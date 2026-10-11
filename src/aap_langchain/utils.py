from typing import Any, Dict

from aap_core.types import ContentType, MediaReference, TokenUsage
from langchain_core.messages.ai import UsageMetadata

# ModelProfile input-modality keys, mapped onto the unified ContentType.
# A profile is a total=False TypedDict, so any key may be absent; absent keys
# stay undecidable rather than being reported as unsupported.
_PROFILE_INPUT_KEYS: Dict[ContentType, str] = {
    "image": "image_inputs",
    "audio": "audio_inputs",
    "video": "video_inputs",
    "document": "pdf_inputs",
}


def token_from_response(usage: UsageMetadata) -> TokenUsage:
    """
    Convert langchain's usage object to a TokenUsage object.

    Args:
        usage (UsageMetadata): The UsageMetadata object to convert.

    Returns:
        TokenUsage: The converted object.
    """
    return TokenUsage(
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        total_tokens=usage["total_tokens"],
    )


def media_ref_to_content_block(ref: MediaReference) -> Dict[str, Any]:
    """
    Convert a resolved MediaRef to a langchain content block.

    Image and audio share one mapping because their block shapes are identical in
    langchain-core 1.x: AudioContentBlock and ImageContentBlock both carry
    type/base64/mime_type/url. Document and video blocks use different fields
    ("file" with a filename, no video block type at all) and will need a branch
    when wired up.

    Two asymmetries between image and audio live below this function, in the
    provider adapters, and cannot be fixed at the block level:

    - Transport: an audio ref with kind="url" is representable here, and
      langchain-core's OpenAI translator rejects it outright
      ("Key base64 is required for audio blocks"), while an image URL passes
      through. Adapters differ, so the block stays canonical and callers who must
      work across providers pass a local path or a data: URI, which resolve_media
      delivers as base64.
    - Format token: the same translator derives OpenAI's input_audio.format from
      the MIME subtype (audio/mpeg becomes "mpeg", which OpenAI rejects; it
      accepts only "wav" and "mp3"). Rewriting mime_type to satisfy one provider
      would break the adapters that read it as a MIME string, so the string is
      passed through unchanged and the mismatch surfaces at the API.

    Args:
        ref (MediaRef): The resolved media reference from BaseLLMChain._media_parts.

    Returns:
        Dict[str, Any]: A langchain content block ready to embed in a message's content list.
    """
    if ref["kind"] == "url":
        return {"type": ref["content_type"], "url": ref["value"]}
    return {
        "type": ref["content_type"],
        "base64": ref["value"],
        "mime_type": ref["mime_type"],
    }


def profile_to_capabilities(profile: Any) -> Dict[ContentType, bool]:
    """Map a langchain model profile onto per-modality input capabilities.

    The profile is the provider-shipped capability table (see the LangChain
    model profiles guide); it is authoritative per model revision, which is
    what makes it preferable to model-family name matching. Only keys actually
    present in the profile produce an opinion.

    Args:
        profile: A ModelProfile mapping, or None/anything non-dict.

    Returns:
        Dict[ContentType, bool]: Modalities the profile asserts an opinion on.
    """
    if not isinstance(profile, dict):
        return {}
    capabilities: Dict[ContentType, bool] = {}
    for content_type, key in _PROFILE_INPUT_KEYS.items():
        value = profile.get(key)
        if isinstance(value, bool):
            capabilities[content_type] = value
    return capabilities


def _ollama_carries_audio() -> bool:
    """Report whether the Ollama chat request schema can carry an audio part.

    This is a fact about the wire format, not about a model family: the message
    type has one field per media kind, so if no audio field exists, no Ollama
    model can receive audio regardless of what its model card claims. Read from
    the installed schema rather than hardcoded, so the answer updates when the
    transport gains one. An unreadable schema reports True, which leaves the
    modality undecidable and keeps auto mode fail-loud.
    """
    try:
        from ollama._types import Message

        fields = set(Message.model_fields)
    except Exception:  # noqa: BLE001 - optional dependency, private module path
        return True
    return any("audio" in field.lower() for field in fields)


def adapter_capabilities(model: Any) -> Dict[ContentType, bool]:
    """Map an adapter's transport onto the modalities it can physically deliver.

    Complements profile_to_capabilities: a model profile states what a model
    understands, this states what the request format can carry, and the second is
    a hard ceiling on the first. Only denials are returned - a transport that
    names an audio field proves nothing about a given model accepting it.

    Args:
        model: A BaseChatModel instance.

    Returns:
        Dict[ContentType, bool]: Modalities the transport cannot deliver, as False.
    """
    for klass in type(model).__mro__:
        if klass.__module__.split(".")[0] == "langchain_ollama":
            if not _ollama_carries_audio():
                return {"audio": False}
            return {}
    return {}


__all__ = [
    "adapter_capabilities",
    "media_ref_to_content_block",
    "profile_to_capabilities",
    "token_from_response",
]
