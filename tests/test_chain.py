"""Offline structural tests for aap_langchain.chain media input support.

No network or model calls: tests exercise _prepare_conversation assembly,
capability detection from model metadata, and the media gate only. The
Hugging Face lookup is stubbed by an autouse fixture so detection is
deterministic offline.
"""

import base64
import logging
from typing import Any, Dict, List, Optional

import pytest
from aap_core.types import AgentMessage, ContentType
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import Field

import aap_langchain.chain as chain_module
from aap_core.utils import (
    extract_repo_id,
    huggingface_capabilities,
    task_tags_to_capabilities,
)
from aap_langchain.chain import ChatCausalMultiTurnsChain
from aap_langchain.utils import (
    _ollama_carries_audio,
    adapter_capabilities,
    media_ref_to_content_block,
    profile_to_capabilities,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# The RIFF form type sits at offset 8, so a valid WAV needs the 12-byte prefix.
WAV_MAGIC = b"RIFF" + (36).to_bytes(4, "little") + b"WAVE"
# An ID3v2 tag with a zero-length body; the MPEG frame sync follows it.
MP3_MAGIC = b"ID3\x04\x00\x00" + b"\x00" * 4 + b"\xff\xfb\x90\x00"

# Capability evidence now comes from metadata, so tests declare it explicitly
# instead of relying on model names.
VISION_PROFILE: Dict[str, bool] = {"image_inputs": True}
TEXT_ONLY_PROFILE: Dict[str, bool] = {"image_inputs": False}
AUDIO_PROFILE: Dict[str, bool] = {"audio_inputs": True}
MULTIMODAL_PROFILE: Dict[str, bool] = {"image_inputs": True, "audio_inputs": True}


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


class TestTransportCeiling:
    """A request format with no field for a modality cannot deliver it to any model.

    This is structural evidence, not a model-name heuristic: the Ollama chat
    message type carries role/content/thinking/images/tool_name/tool_calls and no
    audio, so langchain_ollama raises "Blocks of type audio not supported." on the
    way out. Denying the modality turns that crash into the documented
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

    def test_ollama_transport_still_has_no_audio_field(self):
        """Canary: if this fails, Ollama gained audio and this ceiling must go."""
        assert _ollama_carries_audio() is False


class TestOpenAIBlockContract:
    """Our blocks checked against the translator a provider adapter actually calls.

    Testing the dict shape alone would only prove we agree with ourselves. This
    pins the mapping to langchain-core's consumer, and records which of its
    transports accept audio. The module is internal to langchain-core; if it moves,
    this test fails and the contract needs re-reading, which is the point.
    """

    @staticmethod
    def _translate(block: Dict[str, Any]) -> Any:
        from langchain_core.messages.block_translators.openai import (
            convert_to_openai_data_block,
        )

        return convert_to_openai_data_block(block, api="chat/completions")

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
