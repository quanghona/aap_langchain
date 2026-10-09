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

    The mapping is modality-agnostic for image and audio, whose base64/url
    block shapes are identical in langchain-core 1.x. Document and video
    blocks use different fields and will need a branch when wired up.

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


__all__ = [
    "media_ref_to_content_block",
    "profile_to_capabilities",
    "token_from_response",
]
