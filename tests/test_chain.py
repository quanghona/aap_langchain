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
    media_ref_to_content_block,
    profile_to_capabilities,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Capability evidence now comes from metadata, so tests declare it explicitly
# instead of relying on model names.
VISION_PROFILE: Dict[str, bool] = {"image_inputs": True}
TEXT_ONLY_PROFILE: Dict[str, bool] = {"image_inputs": False}


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
