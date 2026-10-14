"""Offline structural tests for aap_langchain.chain media input support.

No network or model calls: tests exercise _prepare_conversation assembly,
capability detection from model metadata, and the media gate only. The
Hugging Face lookup is stubbed by an autouse fixture so detection is
deterministic offline.
"""

import ast
import base64
import inspect
import logging
import re
import textwrap
from typing import Any, Dict, List, Optional

import pytest
from aap_core.types import AgentMessage, ContentType
from aap_core.utils import (
    extract_repo_id,
    huggingface_capabilities,
    task_tags_to_capabilities,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import Field

import aap_langchain.chain as chain_module
from aap_langchain.chain import ChatCausalMultiTurnsChain
from aap_langchain.utils import (
    _module_carries_blocks,
    _ollama_carries,
    _openai_translator_carries,
    adapter_capabilities,
    media_ref_to_content_block,
    profile_to_capabilities,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# The RIFF form type sits at offset 8, so a valid WAV needs the 12-byte prefix.
WAV_MAGIC = b"RIFF" + (36).to_bytes(4, "little") + b"WAVE"
# An ID3v2 tag with a zero-length body; the MPEG frame sync follows it.
MP3_MAGIC = b"ID3\x04\x00\x00" + b"\x00" * 4 + b"\xff\xfb\x90\x00"
# ISO base media: box size at 0, "ftyp" at 4, major brand at 8. The brand is
# what separates MP4 from QuickTime, so both fixtures differ only there.
MP4_MAGIC = b"\x00\x00\x00\x18" + b"ftyp" + b"isom" + b"\x00" * 4
QT_MAGIC = b"\x00\x00\x00\x18" + b"ftyp" + b"qt  " + b"\x00" * 4

# Capability evidence now comes from metadata, so tests declare it explicitly
# instead of relying on model names.
VISION_PROFILE: Dict[str, bool] = {"image_inputs": True}
TEXT_ONLY_PROFILE: Dict[str, bool] = {"image_inputs": False}
AUDIO_PROFILE: Dict[str, bool] = {"audio_inputs": True}
VIDEO_PROFILE: Dict[str, bool] = {"video_inputs": True}
MULTIMODAL_PROFILE: Dict[str, bool] = {"image_inputs": True, "audio_inputs": True}
TRIMODAL_PROFILE: Dict[str, bool] = {
    "image_inputs": True,
    "audio_inputs": True,
    "video_inputs": True,
}


class FakeChatModel(BaseChatModel):
    """Minimal chat model stub; never invoked by these tests."""

    model_name: str = Field(default="fake-model")

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs,
    ) -> Any:
        raise NotImplementedError  # must not be called in offline tests


class HuggingFaceStub:
    """Stand-in for the model-card lookup: canned results plus call log."""

    def __init__(self) -> None:
        self.results: Dict[str, Dict[ContentType, bool]] = {}
        self.calls: List[str] = []

    def __call__(self, repo_id: str) -> Dict[ContentType, bool]:
        self.calls.append(repo_id)
        return self.results.get(repo_id, {})


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Keep the default lookup inert so no test can reach the Hub."""
    monkeypatch.setattr(chain_module, "huggingface_capabilities", lambda repo_id: {})


@pytest.fixture
def hf(monkeypatch):
    """Install a recording Hugging Face stub over no_network's default."""
    stub = HuggingFaceStub()
    monkeypatch.setattr(chain_module, "huggingface_capabilities", stub)
    return stub


def make_chain(
    model_name: str = "fake-model",
    profile: Optional[Dict[str, bool]] = VISION_PROFILE,
    **kwargs,
) -> ChatCausalMultiTurnsChain:
    return ChatCausalMultiTurnsChain(
        model=FakeChatModel(model_name=model_name, profile=profile),
        system_prompt="You are helpful.",
        user_prompt_template="{query}",
        **kwargs,
    )


def user_message_of(conversation: List[BaseMessage]) -> BaseMessage:
    return conversation[1]  # [0] is the system message


class TestPrepareConversationTextOnly:
    """Regression: text-only behavior must be identical to before media support."""

    def test_plain_text_message(self):
        chain = make_chain()
        conversation = chain._prepare_conversation(AgentMessage(query="hello"))
        assert conversation[0].content == "You are helpful."
        assert user_message_of(conversation).content == "hello"

    def test_template_with_context(self):
        chain = ChatCausalMultiTurnsChain(
            model=FakeChatModel(),
            system_prompt="s",
            user_prompt_template="{query} about {context_topic}",
        )
        msg = AgentMessage(query="q", context={"topic": "t"})
        assert user_message_of(chain._prepare_conversation(msg)).content == "q about t"

    def test_history_appended_after_user_message(self):
        chain = make_chain()
        chain.include_history = 2
        msg = AgentMessage(
            query="now", responses=[("user", "before"), ("agent", "answer")]
        )
        conversation = chain._prepare_conversation(msg)
        assert len(conversation) == 4
        assert conversation[2].content == "before"
        assert conversation[3].content == "answer"


class TestPrepareConversationWithImages:
    """Image media must become native content parts on the current user message."""

    def test_url_image_becomes_content_parts(self):
        chain = make_chain(model_name="gemma3:12b")
        msg = AgentMessage(
            query="what is this?",
            query_media=[("image", "https://example.com/cat.png")],
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert content == [
            {"type": "text", "text": "what is this?"},
            {"type": "image", "url": "https://example.com/cat.png"},
        ]

    def test_local_path_image_becomes_base64_part(self, tmp_path):
        f = tmp_path / "cat.png"
        f.write_bytes(PNG_MAGIC + b"\x00" * 16)
        chain = make_chain(model_name="gemma3:12b")
        msg = AgentMessage(query="describe", query_media=[("image", str(f))])
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert content[0] == {"type": "text", "text": "describe"}
        assert content[1]["type"] == "image"
        assert content[1]["mime_type"] == "image/png"
        assert base64.b64decode(content[1]["base64"]) == f.read_bytes()

    def test_multiple_images_preserve_order(self):
        chain = make_chain(model_name="gemma3:12b")
        msg = AgentMessage(
            query="two?",
            query_media=[
                ("image", "https://example.com/a.png"),
                ("image", "https://example.com/b.jpg"),
            ],
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert [p.get("url", "") for p in content[1:]] == [
            "https://example.com/a.png",
            "https://example.com/b.jpg",
        ]

    def test_no_repr_leak_into_text_part(self):
        """The core leak guard: media must never appear in the interpolated text."""
        chain = make_chain(model_name="gemma3:12b")
        msg = AgentMessage(
            query="what?",
            query_media=[("image", "https://example.com/cat.png")],
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert content[0]["text"] == "what?"
        assert "example.com" not in content[0]["text"]

    def test_images_attach_only_to_current_user_message(self):
        chain = make_chain(model_name="gemma3:12b")
        chain.include_history = 1
        msg = AgentMessage(
            query="new",
            query_media=[("image", "https://example.com/x.png")],
            responses=[("user", "old text turn")],
        )
        conversation = chain._prepare_conversation(msg)
        assert isinstance(user_message_of(conversation).content, list)
        assert conversation[2].content == "old text turn"  # history stays text


class TestPrepareConversationWithAudio:
    """Audio media must reach the model as audio, never as a transcript."""

    def test_url_audio_becomes_content_part(self):
        chain = make_chain(profile=AUDIO_PROFILE)
        msg = AgentMessage(
            query="what song?",
            query_media=[("audio", "https://example.com/clip.wav")],
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert content == [
            {"type": "text", "text": "what song?"},
            {"type": "audio", "url": "https://example.com/clip.wav"},
        ]

    def test_local_path_audio_becomes_base64_part(self, tmp_path):
        f = tmp_path / "clip.wav"
        f.write_bytes(WAV_MAGIC + b"\x00" * 16)
        chain = make_chain(profile=AUDIO_PROFILE)
        msg = AgentMessage(query="listen", query_media=[("audio", str(f))])
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert content[0] == {"type": "text", "text": "listen"}
        assert content[1]["type"] == "audio"
        assert content[1]["mime_type"] == "audio/wav"
        assert base64.b64decode(content[1]["base64"]) == f.read_bytes()

    def test_audio_format_is_sniffed_not_taken_from_the_extension(self, tmp_path):
        """A mislabelled .txt carrying WAV bytes must still deliver audio/wav."""
        f = tmp_path / "clip.txt"
        f.write_bytes(WAV_MAGIC + b"\x00" * 16)
        chain = make_chain(profile=AUDIO_PROFILE)
        msg = AgentMessage(query="listen", query_media=[("audio", str(f))])
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert content[1]["mime_type"] == "audio/wav"

    def test_mp3_tag_is_reported_as_audio_mpeg(self, tmp_path):
        """The MIME string is what an adapter derives its format token from."""
        f = tmp_path / "clip.mp3"
        f.write_bytes(MP3_MAGIC + b"\x00" * 16)
        chain = make_chain(profile=AUDIO_PROFILE)
        msg = AgentMessage(query="listen", query_media=[("audio", str(f))])
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert content[1]["mime_type"] == "audio/mpeg"

    def test_multiple_audio_preserve_order(self):
        chain = make_chain(profile=AUDIO_PROFILE)
        msg = AgentMessage(
            query="two clips",
            query_media=[
                ("audio", "https://example.com/a.mp3"),
                ("audio", "https://example.com/b.wav"),
            ],
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert [block.get("url", "") for block in content[1:]] == [
            "https://example.com/a.mp3",
            "https://example.com/b.wav",
        ]

    def test_images_and_audio_both_attach(self):
        """One message carrying two modalities must lose neither."""
        chain = make_chain(profile=MULTIMODAL_PROFILE)
        msg = AgentMessage(
            query="compare",
            query_media=[
                ("audio", "https://example.com/a.wav"),
                ("image", "https://example.com/b.png"),
            ],
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert [block["type"] for block in content] == ["text", "image", "audio"]

    def test_no_repr_leak_into_text_part(self):
        chain = make_chain(profile=AUDIO_PROFILE)
        msg = AgentMessage(
            query="what?", query_media=[("audio", "https://example.com/a.wav")]
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert content[0]["text"] == "what?"
        assert "example.com" not in content[0]["text"]

    def test_audio_attaches_only_to_current_user_message(self):
        chain = make_chain(profile=AUDIO_PROFILE)
        chain.include_history = 1
        msg = AgentMessage(
            query="new",
            query_media=[("audio", "https://example.com/a.wav")],
            responses=[("user", "old text turn")],
        )
        conversation = chain._prepare_conversation(msg)
        assert isinstance(user_message_of(conversation).content, list)
        assert conversation[2].content == "old text turn"


class TestPrepareConversationWithVideo:
    """Video media must reach the model as one container part, never as frames.

    Frame sampling is the provider's decision (Gemini samples at 1 FPS, other
    APIs expose their own knobs); decomposing a video into N image parts here
    would change what the model was told it received.
    """

    def test_url_video_becomes_content_part(self):
        chain = make_chain(profile=VIDEO_PROFILE)
        msg = AgentMessage(
            query="what happens?",
            query_media=[("video", "https://example.com/clip.mp4")],
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert content == [
            {"type": "text", "text": "what happens?"},
            {"type": "video", "url": "https://example.com/clip.mp4"},
        ]

    def test_local_path_video_becomes_base64_part(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(MP4_MAGIC + b"\x00" * 16)
        chain = make_chain(profile=VIDEO_PROFILE)
        msg = AgentMessage(query="watch", query_media=[("video", str(f))])
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert content[0] == {"type": "text", "text": "watch"}
        assert content[1]["type"] == "video"
        assert content[1]["mime_type"] == "video/mp4"
        assert base64.b64decode(content[1]["base64"]) == f.read_bytes()

    def test_video_format_is_sniffed_not_taken_from_the_extension(self, tmp_path):
        """A mislabelled .txt carrying MP4 bytes must still deliver video/mp4."""
        f = tmp_path / "clip.txt"
        f.write_bytes(MP4_MAGIC + b"\x00" * 16)
        chain = make_chain(profile=VIDEO_PROFILE)
        msg = AgentMessage(query="watch", query_media=[("video", str(f))])
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert content[1]["mime_type"] == "video/mp4"

    def test_quicktime_brand_beats_the_mp4_extension(self, tmp_path):
        """Shipping a .mov as video/mp4 puts the wrong container in front of a provider."""
        f = tmp_path / "clip.mp4"
        f.write_bytes(QT_MAGIC + b"\x00" * 16)
        chain = make_chain(profile=VIDEO_PROFILE)
        msg = AgentMessage(query="watch", query_media=[("video", str(f))])
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert content[1]["mime_type"] == "video/quicktime"

    def test_multiple_videos_preserve_order(self):
        chain = make_chain(profile=VIDEO_PROFILE)
        msg = AgentMessage(
            query="two clips",
            query_media=[
                ("video", "https://example.com/a.mp4"),
                ("video", "https://example.com/b.mov"),
            ],
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert [block.get("url", "") for block in content[1:]] == [
            "https://example.com/a.mp4",
            "https://example.com/b.mov",
        ]

    def test_images_audio_and_video_all_attach(self):
        """One message carrying three modalities must lose none."""
        chain = make_chain(profile=TRIMODAL_PROFILE)
        msg = AgentMessage(
            query="compare",
            query_media=[
                ("video", "https://example.com/c.mp4"),
                ("audio", "https://example.com/a.wav"),
                ("image", "https://example.com/b.png"),
            ],
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert [block["type"] for block in content] == [
            "text",
            "image",
            "audio",
            "video",
        ]

    def test_no_repr_leak_into_text_part(self):
        chain = make_chain(profile=VIDEO_PROFILE)
        msg = AgentMessage(
            query="what?", query_media=[("video", "https://example.com/a.mp4")]
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert content[0]["text"] == "what?"
        assert "example.com" not in content[0]["text"]

    def test_video_attaches_only_to_current_user_message(self):
        chain = make_chain(profile=VIDEO_PROFILE)
        chain.include_history = 1
        msg = AgentMessage(
            query="new",
            query_media=[("video", "https://example.com/a.mp4")],
            responses=[("user", "old text turn")],
        )
        conversation = chain._prepare_conversation(msg)
        assert isinstance(user_message_of(conversation).content, list)
        assert conversation[2].content == "old text turn"


class TestCapabilityDetection:
    """Capabilities come from model metadata, never from the model name."""

    def test_profile_reports_image_support(self):
        chain = make_chain(profile={"image_inputs": True})
        assert chain.detect_capabilities() == {"image": True}

    def test_profile_reports_image_absence(self):
        chain = make_chain(profile={"image_inputs": False})
        assert chain.detect_capabilities() == {"image": False}

    def test_profile_reports_every_modality(self):
        chain = make_chain(
            profile={
                "image_inputs": True,
                "audio_inputs": False,
                "video_inputs": False,
                "pdf_inputs": True,
            }
        )
        assert chain.detect_capabilities() == {
            "image": True,
            "audio": False,
            "video": False,
            "document": True,
        }

    def test_family_name_alone_is_not_evidence(self):
        """A vision-looking name without metadata stays undecidable (fail loud)."""
        assert (
            make_chain(
                model_name="gemma4-audio-26b", profile=None
            ).detect_capabilities()
            == {}
        )

    def test_huggingface_fills_gaps_left_by_a_partial_profile(self, hf):
        hf.results["google/gemma-3n-e4b-it"] = {"audio": True, "video": False}
        chain = make_chain(
            model_name="google/gemma-3n-e4b-it",
            profile={"image_inputs": True},
        )
        assert chain.detect_capabilities() == {
            "image": True,
            "audio": True,
            "video": False,
        }

    def test_huggingface_not_consulted_when_profile_is_complete(self, hf):
        chain = make_chain(
            model_name="org/repo",
            profile={
                "image_inputs": True,
                "audio_inputs": True,
                "video_inputs": True,
                "pdf_inputs": True,
            },
        )
        chain.detect_capabilities()
        assert hf.calls == []

    def test_variants_of_one_family_report_different_capabilities(self, hf):
        """The bug name matching had: same prefix, different modality per size."""
        hf.results["google/gemma-3n-e4b-it"] = task_tags_to_capabilities(
            ["image-text-to-text", "audio-text-to-text"]
        )
        hf.results["google/gemma-3-27b-it"] = task_tags_to_capabilities(
            ["image-text-to-text"]
        )
        small = make_chain(model_name="google/gemma-3n-e4b-it", profile=None)
        large = make_chain(model_name="google/gemma-3-27b-it", profile=None)
        assert small.detect_capabilities()["audio"] is True
        assert large.detect_capabilities()["audio"] is False

    def test_provider_names_are_not_looked_up_on_huggingface(self, hf):
        make_chain(model_name="gpt-4o", profile=None).detect_capabilities()
        make_chain(model_name="gemma3:12b", profile=None).detect_capabilities()
        assert hf.calls == []

    def test_capabilities_cached_per_model_instance(self, hf):
        hf.results["org/repo"] = {"image": True}
        chain = make_chain(model_name="org/repo", profile=None)
        chain.detect_capabilities()
        chain.detect_capabilities()
        assert hf.calls == ["org/repo"]

    def test_cache_invalidated_when_model_is_swapped(self, hf):
        chain = make_chain(model_name="org/a", profile=None)
        chain.detect_capabilities()
        chain.model = FakeChatModel(model_name="org/b")
        chain.detect_capabilities()
        assert hf.calls == ["org/a", "org/b"]


class TestCapabilityMapping:
    """Pure metadata mappers used by detection."""

    def test_profile_ignores_non_bool_values(self):
        assert profile_to_capabilities({"image_inputs": "yes"}) == {}

    def test_profile_ignores_output_and_infra_keys(self):
        assert (
            profile_to_capabilities({"image_outputs": True, "tool_calling": True}) == {}
        )

    def test_profile_rejects_non_dict(self):
        assert profile_to_capabilities(None) == {}

    def test_task_tag_reads_only_the_input_side(self):
        """text-to-speech generates audio; it does not accept it."""
        assert task_tags_to_capabilities(["text-to-speech"])["audio"] is False
        assert task_tags_to_capabilities(["audio-text-to-text"])["audio"] is True

    def test_any_to_any_accepts_everything(self):
        assert task_tags_to_capabilities(["any-to-any"]) == {
            "image": True,
            "audio": True,
            "video": True,
            "document": True,
        }

    def test_document_is_never_denied_from_absence(self):
        """Task tags do not express PDF support, so it stays undecidable."""
        capabilities = task_tags_to_capabilities(["image-text-to-text"])
        assert "document" not in capabilities

    def test_without_task_tags_nothing_is_claimed(self):
        assert task_tags_to_capabilities([]) == {}
        assert task_tags_to_capabilities([None, ""]) == {}

    @pytest.mark.parametrize(
        "candidate,expected",
        [
            ("google/gemma-3-4b-it", "google/gemma-3-4b-it"),
            ("org/repo:q8_0", "org/repo"),
            ("org/repo@rev", "org/repo"),
            ("huggingface:org/repo", "org/repo"),
            ("gpt-4o", None),
            ("gemma3:12b", None),
            (
                "mistralai/Mistral-7B-Instruct-v0.3",
                "mistralai/Mistral-7B-Instruct-v0.3",
            ),
        ],
    )
    def test_extract_repo_id(self, candidate, expected):
        assert extract_repo_id([candidate]) == expected

    def test_huggingface_lookup_failure_is_undecidable(self, monkeypatch):
        import huggingface_hub

        def boom(*args, **kwargs):
            raise RuntimeError("offline")

        monkeypatch.setattr(huggingface_hub, "model_info", boom)
        assert huggingface_capabilities("org/repo") == {}

    def test_missing_hub_extra_warns_once_and_stays_undecidable(
        self, monkeypatch, caplog
    ):
        """Without the optional dependency, detection degrades to fail-loud."""
        import sys

        import aap_core.utils as core_utils

        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        monkeypatch.setattr(core_utils, "_HUB_UNAVAILABLE_LOGGED", False)
        with caplog.at_level(logging.WARNING, logger="aap_core.utils"):
            assert huggingface_capabilities("org/repo") == {}
            assert huggingface_capabilities("org/repo") == {}
        assert len(caplog.records) == 1

    def test_huggingface_reads_task_tags_from_the_card(self, monkeypatch):
        import huggingface_hub

        class Info:
            pipeline_tag = "image-text-to-text"
            tags = ["transformers", "audio-text-to-text", "license:apache-2.0"]

        monkeypatch.setattr(huggingface_hub, "model_info", lambda repo_id: Info())
        assert huggingface_capabilities("org/repo") == {
            "image": True,
            "audio": True,
            "video": False,
        }


class TestGateIntegration:
    """End-to-end of the core gate through the langchain funnel."""

    def test_text_only_model_drops_images_with_warning(self, caplog):
        chain = make_chain(model_name="llama3.2:latest", profile=TEXT_ONLY_PROFILE)
        msg = AgentMessage(
            query="look", query_media=[("image", "https://example.com/a.png")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            content = user_message_of(chain._prepare_conversation(msg)).content
        assert content == "look"  # degraded to plain text, no exception
        assert "does not support image input" in caplog.text

    def test_disabled_policy_overrides_vision_detection(self):
        chain = make_chain(model_name="gemma3:12b", media_support={"image": "disabled"})
        msg = AgentMessage(
            query="look", query_media=[("image", "https://example.com/a.png")]
        )
        assert user_message_of(chain._prepare_conversation(msg)).content == "look"

    def test_enabled_policy_forces_parts_on_unknown_model(self):
        chain = make_chain(
            model_name="some-brand-new-model",
            profile=None,
            media_support={"image": "enabled"},
        )
        msg = AgentMessage(
            query="look", query_media=[("image", "https://example.com/a.png")]
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert isinstance(content, list) and content[1]["type"] == "image"

    def test_undecidable_model_attaches_images_fail_loud(self):
        """Unknown model + image media: attach and let the API error surface."""
        chain = make_chain(model_name="some-brand-new-model", profile=None)
        msg = AgentMessage(
            query="look", query_media=[("image", "https://example.com/a.png")]
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert isinstance(content, list)

    def test_missing_local_image_raises(self, tmp_path):
        chain = make_chain(model_name="gemma3:12b")
        msg = AgentMessage(
            query="look", query_media=[("image", str(tmp_path / "nope.png"))]
        )
        with pytest.raises(FileNotFoundError):
            chain._prepare_conversation(msg)


class TestMediaRefToContentBlock:
    """Direct tests of the MediaRef -> langchain block mapping."""

    def test_url_ref(self):
        block = media_ref_to_content_block(
            {
                "content_type": "image",
                "kind": "url",
                "value": "http://x/a",
                "mime_type": "image/png",
            }
        )
        assert block == {"type": "image", "url": "http://x/a"}

    def test_base64_ref(self):
        block = media_ref_to_content_block(
            {
                "content_type": "image",
                "kind": "base64",
                "value": "AAAA",
                "mime_type": "image/jpeg",
            }
        )
        assert block == {
            "type": "image",
            "base64": "AAAA",
            "mime_type": "image/jpeg",
        }

    def test_audio_ref_same_shape(self):
        """Audio blocks share the base64 shape; mapping is modality-generic."""
        block = media_ref_to_content_block(
            {
                "content_type": "audio",
                "kind": "base64",
                "value": "AAAA",
                "mime_type": "audio/wav",
            }
        )
        assert block["type"] == "audio"

    def test_audio_url_ref(self):
        """URL refs stay canonical here; whether an adapter takes them is below."""
        block = media_ref_to_content_block(
            {
                "content_type": "audio",
                "kind": "url",
                "value": "https://x/a.wav",
                "mime_type": "audio/wav",
            }
        )
        assert block == {"type": "audio", "url": "https://x/a.wav"}

    def test_video_ref_same_shape(self):
        """Video blocks share the base64 shape; mapping is modality-generic."""
        block = media_ref_to_content_block(
            {
                "content_type": "video",
                "kind": "base64",
                "value": "AAAA",
                "mime_type": "video/mp4",
            }
        )
        assert block == {
            "type": "video",
            "base64": "AAAA",
            "mime_type": "video/mp4",
        }

    def test_video_url_ref(self):
        block = media_ref_to_content_block(
            {
                "content_type": "video",
                "kind": "url",
                "value": "https://x/a.mp4",
                "mime_type": "video/mp4",
            }
        )
        assert block == {"type": "video", "url": "https://x/a.mp4"}


class TestGatePerModality:
    """The gate is per modality: audio support says nothing about image support."""

    def test_audio_denied_by_profile_drops_with_warning(self, caplog):
        chain = make_chain(profile={"audio_inputs": False})
        msg = AgentMessage(
            query="listen", query_media=[("audio", "https://example.com/a.wav")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            content = user_message_of(chain._prepare_conversation(msg)).content
        assert content == "listen"
        assert "does not support audio input" in caplog.text

    def test_disabled_policy_is_not_applied_to_other_modalities(self):
        chain = make_chain(
            profile=None,
            media_support={"image": "enabled", "audio": "disabled"},
        )
        msg = AgentMessage(
            query="both",
            query_media=[
                ("image", "https://example.com/a.png"),
                ("audio", "https://example.com/b.wav"),
            ],
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert [block["type"] for block in content] == ["text", "image"]

    def test_enabled_policy_forces_audio_parts_on_unknown_model(self):
        chain = make_chain(
            model_name="some-brand-new-model",
            profile=None,
            media_support={"audio": "enabled"},
        )
        msg = AgentMessage(
            query="listen", query_media=[("audio", "https://example.com/a.wav")]
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert isinstance(content, list) and content[1]["type"] == "audio"

    def test_undecidable_audio_attaches_fail_loud(self):
        chain = make_chain(model_name="some-brand-new-model", profile=None)
        msg = AgentMessage(
            query="listen", query_media=[("audio", "https://example.com/a.wav")]
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert isinstance(content, list)

    def test_missing_local_audio_raises(self, tmp_path):
        chain = make_chain(profile=AUDIO_PROFILE)
        msg = AgentMessage(
            query="listen", query_media=[("audio", str(tmp_path / "nope.wav"))]
        )
        with pytest.raises(FileNotFoundError):
            chain._prepare_conversation(msg)

    def test_video_denied_by_profile_drops_with_warning(self, caplog):
        chain = make_chain(profile={"video_inputs": False})
        msg = AgentMessage(
            query="watch", query_media=[("video", "https://example.com/a.mp4")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            content = user_message_of(chain._prepare_conversation(msg)).content
        assert content == "watch"
        assert "does not support video input" in caplog.text

    def test_disabled_video_policy_is_not_applied_to_other_modalities(self):
        chain = make_chain(
            profile=None,
            media_support={"image": "enabled", "video": "disabled"},
        )
        msg = AgentMessage(
            query="both",
            query_media=[
                ("image", "https://example.com/a.png"),
                ("video", "https://example.com/b.mp4"),
            ],
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert [block["type"] for block in content] == ["text", "image"]

    def test_enabled_policy_forces_video_parts_on_unknown_model(self):
        chain = make_chain(
            model_name="some-brand-new-model",
            profile=None,
            media_support={"video": "enabled"},
        )
        msg = AgentMessage(
            query="watch", query_media=[("video", "https://example.com/a.mp4")]
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert isinstance(content, list) and content[1]["type"] == "video"

    def test_undecidable_video_attaches_fail_loud(self):
        chain = make_chain(model_name="some-brand-new-model", profile=None)
        msg = AgentMessage(
            query="watch", query_media=[("video", "https://example.com/a.mp4")]
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert isinstance(content, list)

    def test_missing_local_video_raises(self, tmp_path):
        chain = make_chain(profile=VIDEO_PROFILE)
        msg = AgentMessage(
            query="watch", query_media=[("video", str(tmp_path / "nope.mp4"))]
        )
        with pytest.raises(FileNotFoundError):
            chain._prepare_conversation(msg)


def make_ollama_chain(
    model_name: str = "llama3.2", **kwargs
) -> ChatCausalMultiTurnsChain:
    """Chain over the real Ollama adapter; constructed offline, never invoked."""
    langchain_ollama = pytest.importorskip("langchain_ollama")
    return ChatCausalMultiTurnsChain(
        model=langchain_ollama.ChatOllama(model=model_name),
        system_prompt="You are helpful.",
        user_prompt_template="{query}",
        **kwargs,
    )


class FakeOpenAIChatModel(FakeChatModel):
    """Stand-in for the langchain_openai module prefix; the package is not installed.

    adapter_capabilities dispatches on module root, so overriding __module__
    exercises that branch without an API key. The ceiling itself reads
    langchain_core's translator, which is installed.
    """


FakeOpenAIChatModel.__module__ = "langchain_openai.chat_models"


class TestTransportCeiling:
    """A request format with no field for a modality cannot deliver it to any model.

    This is structural evidence, not a model-name heuristic. The Ollama chat
    message type carries role/content/thinking/images/tool_name/tool_calls and no
    audio or video, so langchain_ollama silently drops a video block on the
    AIMessage path and raises "Blocks of type video not supported." on the other;
    langchain_openai's translator raises on every video block, on either API.
    Denying the modality turns a silent drop or a crash into the documented
    degrade-to-text behaviour, which is what auto mode is for.
    """

    def test_ollama_denies_audio_without_any_metadata_source(self):
        assert make_ollama_chain().detect_capabilities().get("audio") is False

    def test_ollama_audio_media_degrades_to_text(self, caplog):
        chain = make_ollama_chain()
        msg = AgentMessage(
            query="listen", query_media=[("audio", "https://example.com/a.wav")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            content = user_message_of(chain._prepare_conversation(msg)).content
        assert content == "listen"
        assert "does not support audio input" in caplog.text

    def test_ollama_ceiling_is_per_modality(self):
        """Denying audio must not deny images, which the transport does carry."""
        chain = make_ollama_chain()
        msg = AgentMessage(
            query="look", query_media=[("image", "https://example.com/a.png")]
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert isinstance(content, list) and content[1]["type"] == "image"

    def test_transport_ceiling_overrides_a_model_card_affirmation(self, hf):
        """The wire format is a hard limit, so it wins over a positive card tag."""
        hf.results["google/gemma-3n-e4b-it"] = task_tags_to_capabilities(
            ["image-text-to-text", "audio-text-to-text"]
        )
        chain = make_ollama_chain(model_name="google/gemma-3n-e4b-it")
        capabilities = chain.detect_capabilities()
        assert capabilities["audio"] is False
        assert capabilities["image"] is True

    def test_other_adapters_get_no_opinion(self):
        assert adapter_capabilities(FakeChatModel()) == {}

    def test_explicit_enable_overrides_the_ceiling(self):
        """An enabled policy outranks detection, so the adapter error must surface."""
        chain = make_ollama_chain(media_support={"audio": "enabled"})
        msg = AgentMessage(
            query="listen", query_media=[("audio", "https://example.com/a.wav")]
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert isinstance(content, list) and content[1]["type"] == "audio"

    def test_ollama_transport_still_has_no_audio_or_video_field(self):
        """Canary: if this fails, Ollama gained a field and that ceiling must go."""
        assert _ollama_carries("audio") is False
        assert _ollama_carries("video") is False
        assert _ollama_carries("image") is True  # images field exists

    def test_ollama_denies_video_without_any_metadata_source(self):
        assert make_ollama_chain().detect_capabilities().get("video") is False

    def test_ollama_video_media_degrades_to_text(self, caplog):
        chain = make_ollama_chain()
        msg = AgentMessage(
            query="watch", query_media=[("video", "https://example.com/a.mp4")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            content = user_message_of(chain._prepare_conversation(msg)).content
        assert content == "watch"
        assert "does not support video input" in caplog.text

    def test_ollama_video_denial_leaves_images_attached(self):
        """One denial must not become a blanket: images the transport does carry."""
        chain = make_ollama_chain()
        msg = AgentMessage(
            query="look and watch",
            query_media=[
                ("image", "https://example.com/a.png"),
                ("video", "https://example.com/b.mp4"),
            ],
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert [block["type"] for block in content] == ["text", "image"]

    def test_explicit_enable_overrides_the_video_ceiling(self):
        chain = make_ollama_chain(media_support={"video": "enabled"})
        msg = AgentMessage(
            query="watch", query_media=[("video", "https://example.com/a.mp4")]
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert isinstance(content, list) and content[1]["type"] == "video"

    def test_openai_family_denies_video_by_translator_probe(self):
        """The denial follows the module prefix, so Azure inherits it too."""
        assert adapter_capabilities(FakeOpenAIChatModel()) == {"video": False}

    def test_openai_family_video_media_degrades_to_text(self, caplog):
        chain = ChatCausalMultiTurnsChain(
            model=FakeOpenAIChatModel(),
            system_prompt="You are helpful.",
            user_prompt_template="{query}",
        )
        msg = AgentMessage(
            query="watch", query_media=[("video", "https://example.com/a.mp4")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            content = user_message_of(chain._prepare_conversation(msg)).content
        assert content == "watch"
        assert "does not support video input" in caplog.text

    def test_openai_translator_still_rejects_video(self):
        """Canary: if this fails, the translator gained video and the denial lifts.

        Image and audio assertions keep the probe honest in the other direction:
        a probe that denied everything would pass the video check alone.
        """
        assert _openai_translator_carries("video") is False
        assert _openai_translator_carries("image") is True
        assert _openai_translator_carries("audio") is True


class TestOpenAIBlockContract:
    """Our blocks checked against the translator a provider adapter actually calls.

    Testing the dict shape alone would only prove we agree with ourselves. This
    pins the mapping to langchain-core's consumer, and records which of its
    transports accept audio. The module is internal to langchain-core; if it moves,
    this test fails and the contract needs re-reading, which is the point.
    """

    @staticmethod
    def _translate(block: Dict[str, Any], api: str = "chat/completions") -> Any:
        from langchain_core.messages.block_translators.openai import (
            convert_to_openai_data_block,
        )

        return convert_to_openai_data_block(block, api=api)

    def test_base64_audio_reaches_openai_as_input_audio(self, tmp_path):
        f = tmp_path / "clip.wav"
        f.write_bytes(WAV_MAGIC + b"\x00" * 16)
        chain = make_chain(profile=AUDIO_PROFILE)
        msg = AgentMessage(query="listen", query_media=[("audio", str(f))])
        block = user_message_of(chain._prepare_conversation(msg)).content[1]
        assert self._translate(block) == {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(f.read_bytes()).decode(),
                "format": "wav",
            },
        }

    def test_url_audio_is_rejected_by_the_openai_translator(self):
        """Documented limitation: audio URLs are not portable across adapters."""
        chain = make_chain(profile=AUDIO_PROFILE)
        msg = AgentMessage(
            query="listen", query_media=[("audio", "https://example.com/a.wav")]
        )
        block = user_message_of(chain._prepare_conversation(msg)).content[1]
        with pytest.raises(ValueError, match="base64 is required"):
            self._translate(block)

    def test_audio_format_token_comes_from_the_mime_subtype(self):
        """audio/mpeg becomes "mpeg", which OpenAI's wav|mp3 enum rejects."""
        assert (
            self._translate(
                {"type": "audio", "base64": "AAAA", "mime_type": "audio/mpeg"}
            )["input_audio"]["format"]
            == "mpeg"
        )

    def test_video_blocks_are_rejected_on_either_api(self):
        """Why the OpenAI-family ceiling exists: no video field on either wire."""
        for api in ("chat/completions", "responses"):
            with pytest.raises(ValueError, match="not supported"):
                self._translate(
                    {"type": "video", "base64": "AAAA", "mime_type": "video/mp4"},
                    api=api,
                )

    def test_our_video_block_is_rejected_by_the_openai_translator(self):
        """The block the funnel builds, fed to the translator as the adapter would."""
        chain = make_chain(profile=VIDEO_PROFILE)
        msg = AgentMessage(
            query="watch", query_media=[("video", "https://example.com/a.mp4")]
        )
        block = user_message_of(chain._prepare_conversation(msg)).content[1]
        with pytest.raises(ValueError, match="not supported"):
            self._translate(block)


class TestGeminiVideoContract:
    """Our blocks checked against the converter langchain_google_genai sends them to.

    Gemini needs no ceiling and no mapping branch: its adapter routes every
    data content block (is_data_content_block covers image, video and audio)
    through its own part converter, which reads the same base64/mime_type/url
    fields media_ref_to_content_block produces. These tests pin that agreement
    so a block-shape change on either side fails here, not at the API.
    """

    @staticmethod
    def _gemini_model():
        langchain_google_genai = pytest.importorskip("langchain_google_genai")
        return langchain_google_genai.ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", google_api_key="offline-test"
        )

    def test_gemini_gets_no_ceiling_opinion(self):
        """The transport carries video; adapter_capabilities must stay silent."""
        assert adapter_capabilities(self._gemini_model()) == {}

    def test_our_video_block_is_a_data_content_block(self):
        """The dispatch key Gemini's converter branches on, both ref kinds."""
        from langchain_core.messages.content import is_data_content_block

        assert is_data_content_block({"type": "video", "url": "https://x/a.mp4"})
        assert is_data_content_block(
            {"type": "video", "base64": "AAAA", "mime_type": "video/mp4"}
        )

    def test_funnel_output_reaches_gemini_inline_part(self, tmp_path):
        """End to end offline: local file -> funnel -> Gemini Part, byte-exact."""
        from langchain_google_genai.chat_models import _convert_to_parts

        f = tmp_path / "clip.mp4"
        f.write_bytes(MP4_MAGIC + b"\x00" * 16)
        chain = ChatCausalMultiTurnsChain(
            model=self._gemini_model(),
            system_prompt="You are helpful.",
            user_prompt_template="{query}",
        )
        msg = AgentMessage(query="watch", query_media=[("video", str(f))])
        blocks = user_message_of(chain._prepare_conversation(msg)).content
        parts = _convert_to_parts(blocks)
        assert parts[0].text == "watch"
        assert parts[1].inline_data.mime_type == "video/mp4"
        assert parts[1].inline_data.data == f.read_bytes()


class TestRealOpenAIDispatch:
    """The langchain_openai ceiling against the real adapter, now installed.

    The FakeOpenAIChatModel stand-in pins the dispatch mechanism; these pin the
    production path, including the finding that a custom base_url does not
    rescue video: the adapter translates every data block before the request
    reaches any endpoint, so a DashScope-compatible ChatOpenAI fails in the
    adapter exactly like api.openai.com does.
    """

    @staticmethod
    def _openai_model(**kwargs):
        langchain_openai = pytest.importorskip("langchain_openai")
        return langchain_openai.ChatOpenAI(api_key="offline-test", **kwargs)

    def test_real_chatopenai_denies_video(self):
        assert adapter_capabilities(self._openai_model(model="gpt-4o")) == {
            "video": False
        }

    def test_custom_base_url_still_denies_video(self):
        """Canary for the rejected base_url refinement: compat endpoints crash too."""
        model = self._openai_model(
            model="qwen-vl-max",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        assert adapter_capabilities(model) == {"video": False}

    def test_real_chatopenai_video_degrades_to_text(self, caplog):
        chain = ChatCausalMultiTurnsChain(
            model=self._openai_model(model="gpt-4o"),
            system_prompt="You are helpful.",
            user_prompt_template="{query}",
        )
        msg = AgentMessage(
            query="watch", query_media=[("video", "https://example.com/a.mp4")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            content = user_message_of(chain._prepare_conversation(msg)).content
        assert content == "watch"
        assert "does not support video input" in caplog.text

    def test_real_chatopenai_keeps_image_and_audio_undenied(self):
        """The probe must deny only what the translator refuses."""
        capabilities = adapter_capabilities(self._openai_model(model="gpt-4o"))
        assert "image" not in capabilities
        assert "audio" not in capabilities


class FakeHuggingFaceChatModel(FakeChatModel):
    """Stand-in for the langchain_huggingface root; its converter passes content raw.

    The probe reads the module the class claims, so pointing __module__ at the
    installed huggingface.py exercises the production path without a model
    download. The module must be imported first: inspect resolves a class's
    module through sys.modules.
    """


FakeHuggingFaceChatModel.__module__ = "langchain_huggingface.chat_models.huggingface"


class FakeTongyiChatModel(FakeChatModel):
    """Stand-in for the langchain_community root (ChatTongyi), same raw path."""


FakeTongyiChatModel.__module__ = "langchain_community.chat_models.tongyi"


class TestRawContentTransportCeiling:
    """Adapters that serialize message.content verbatim cannot carry media parts.

    Verified in the installed sources: neither langchain_huggingface's
    _convert_message_to_dict nor ChatTongyi's conversion names a block
    translator or a media part shape, so a content-block list reaches the wire
    unchanged and dies at the API whatever the model behind it understands.
    They get a deny-all ceiling like any other transport with no field for the
    modality - and the same escape hatch: an explicit media_support declaration
    bypasses detection, so the adapter's own error surfaces.
    """

    @staticmethod
    def _chain(model: Any, **kwargs) -> ChatCausalMultiTurnsChain:
        return ChatCausalMultiTurnsChain(
            model=model,
            system_prompt="You are helpful.",
            user_prompt_template="{query}",
            **kwargs,
        )

    def test_huggingface_root_denies_every_media_modality(self):
        pytest.importorskip("langchain_huggingface")
        import langchain_huggingface.chat_models.huggingface  # noqa: F401

        assert adapter_capabilities(FakeHuggingFaceChatModel()) == {
            "image": False,
            "audio": False,
            "video": False,
        }

    def test_tongyi_root_denies_every_media_modality(self):
        import langchain_community.chat_models.tongyi  # noqa: F401

        assert adapter_capabilities(FakeTongyiChatModel()) == {
            "image": False,
            "audio": False,
            "video": False,
        }

    def test_raw_content_video_degrades_to_text(self, caplog):
        pytest.importorskip("langchain_huggingface")
        import langchain_huggingface.chat_models.huggingface  # noqa: F401

        chain = self._chain(FakeHuggingFaceChatModel())
        msg = AgentMessage(
            query="watch", query_media=[("video", "https://example.com/a.mp4")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            content = user_message_of(chain._prepare_conversation(msg)).content
        assert content == "watch"
        assert "does not support video input" in caplog.text

    def test_enabled_policy_overrides_the_raw_content_denial(self):
        pytest.importorskip("langchain_huggingface")
        import langchain_huggingface.chat_models.huggingface  # noqa: F401

        chain = self._chain(
            FakeHuggingFaceChatModel(), media_support={"video": "enabled"}
        )
        msg = AgentMessage(
            query="watch", query_media=[("video", "https://example.com/a.mp4")]
        )
        content = user_message_of(chain._prepare_conversation(msg)).content
        assert isinstance(content, list) and content[1]["type"] == "video"

    def test_marker_probe_agrees_with_the_installed_sources(self):
        """Canary: if either module gains block handling, the deny-all must lift."""
        import langchain_community.chat_models.tongyi  # noqa: F401
        import langchain_huggingface.chat_models.huggingface  # noqa: F401

        assert _module_carries_blocks(FakeHuggingFaceChatModel()) is False
        assert _module_carries_blocks(FakeTongyiChatModel()) is False

    def test_community_adapter_that_names_media_parts_is_not_denied(self):
        """The probe reads the module, not the root: a community adapter whose
        code does name media part shapes stays undecidable, not blanket-denied."""
        import langchain_community.chat_models.ollama  # noqa: F401

        class FakeCommunityOllama(FakeChatModel):
            pass

        FakeCommunityOllama.__module__ = "langchain_community.chat_models.ollama"
        assert _module_carries_blocks(FakeCommunityOllama()) is True
        assert adapter_capabilities(FakeCommunityOllama()) == {}

    def test_unreadable_module_stays_undecidable(self):
        """A class whose module cannot be resolved reports carried: fail loud."""

        class FakeFrozenChatModel(FakeChatModel):
            pass

        FakeFrozenChatModel.__module__ = "not_a_real_package.chat_models"
        assert _module_carries_blocks(FakeFrozenChatModel()) is True
        assert adapter_capabilities(FakeFrozenChatModel()) == {}


class TestMappingIsProviderAgnostic:
    """The bridge invariant: media mapping is a function of the reference alone.

    Provider knowledge belongs in profile_to_capabilities and
    adapter_capabilities, never in the mapper. The frameworks ship hundreds of
    provider packages; a mapper that branches on model identity re-scatters
    per-provider special cases across the integration and cannot scale. These
    tests pin the structure so a future session cannot regress it silently.
    """

    @staticmethod
    def _body_source(fn) -> str:
        """The function body as code, docstring removed: prose may name providers."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        definition = tree.body[0]
        definition.body = [
            statement
            for statement in definition.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
        ]
        return ast.unparse(definition)

    def test_mapper_takes_only_the_reference(self):
        """No model parameter means no way to branch on model identity."""
        assert list(inspect.signature(media_ref_to_content_block).parameters) == ["ref"]

    def test_mapper_body_names_no_provider_surface(self):
        body = self._body_source(media_ref_to_content_block)
        assert not re.search(r"\bmodel\b|\bprofile\b|__module__|adapter", body)
