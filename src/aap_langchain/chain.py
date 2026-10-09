from collections.abc import Callable, Sequence
from typing import Any, Dict, List, Tuple
from aap_core import utils
from aap_core.chain import BaseCausalMultiTurnsChain
from aap_core.types import AgentMessage, ContentType, TokenUsage
from aap_core.utils import extract_repo_id, huggingface_capabilities
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

from aap_langchain.utils import (
    media_ref_to_content_block,
    profile_to_capabilities,
    token_from_response,
)

# Modalities the media gate can consult; detection reports on these.
_GATED_MODALITIES: Tuple[ContentType, ...] = ("image", "audio", "video", "document")


class ChatCausalMultiTurnsChain(BaseCausalMultiTurnsChain[BaseMessage, AIMessage]):
    # For resuability at runtime, wen need to control the assignment and rebuild
    # the model with its partners to operate correctly
    _model: BaseChatModel = PrivateAttr()
    _system_prompt: str = PrivateAttr()
    _user_prompt_template: str = PrivateAttr()
    _tool_dict: Dict[str, Callable | BaseTool] = PrivateAttr({})
    _tool_choice: str | None = PrivateAttr()
    _chain = PrivateAttr()
    _capabilities_cache: Dict[ContentType, bool] | None = PrivateAttr(default=None)

    def __init__(
        self,
        model: BaseChatModel,
        system_prompt: str,
        user_prompt_template: str = "{query}",
        tools: Sequence[Callable | BaseTool] = [],
        tool_choice: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model = model
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template
        self.bind_tools(tools, tool_choice=tool_choice)

    def detect_capabilities(self) -> Dict[ContentType, bool]:
        """Resolve per-modality input support from model metadata, not model names.

        Sources, in priority order:
        1. What the user declared in media_support ("enabled"/"disabled"). The
           core gate applies that before this method runs, so an explicit
           declaration always wins.
        2. The langchain model profile (model.profile), the capability table the
           provider ships for the exact model revision.
        3. Hugging Face model-card metadata, for models identified by repo id
           (needs the "hf" extra).

        Model-family names carry no weight here: variants of one family differ
        per modality (an audio-capable small model and a larger sibling without
        audio input share a name prefix), so name matching misreports exactly
        the cases that matter. Modalities no source reports on stay absent, and
        auto mode treats them as capable (fail loud).

        The result is cached per model instance and invalidated when the model
        is replaced, since the Hugging Face source costs a network request.
        """
        if self._capabilities_cache is None:
            self._capabilities_cache = self._resolve_capabilities()
        return dict(self._capabilities_cache)

    def _resolve_capabilities(self) -> Dict[ContentType, bool]:
        """Query the metadata sources once, filling gaps from the next source."""
        capabilities = profile_to_capabilities(getattr(self._model, "profile", None))
        if set(capabilities) != set(_GATED_MODALITIES):
            repo_id = extract_repo_id(self._identifier_candidates())
            if repo_id is not None:
                for modality, value in huggingface_capabilities(repo_id).items():
                    capabilities.setdefault(modality, value)
        return capabilities

    def _identifier_candidates(self) -> List[str]:
        """Model identifiers to test for a Hugging Face repo id, best first."""
        profile = getattr(self._model, "profile", None)
        candidates: List[str] = []
        if isinstance(profile, dict) and isinstance(profile.get("name"), str):
            candidates.append(profile["name"])
        try:
            name = self._model._get_ls_params().get("ls_model_name")
            if isinstance(name, str):
                candidates.append(name)
        except Exception:  # noqa: BLE001 - not all integrations implement it
            pass
        for holder in (self._model, getattr(self._model, "llm", None)):
            if holder is None:
                continue
            for attr in ("repo_id", "model_id", "model_name", "model"):
                value = getattr(holder, attr, None)
                if isinstance(value, str):
                    candidates.append(value)
        return candidates

    def _prepare_conversation(self, message: AgentMessage) -> List[BaseMessage]:
        user_prompt = self._user_prompt_template.format(**message.format_kwargs())
        images = self._image_parts(message)
        user_content: str | List[Any] = user_prompt
        if images:
            user_content = [
                {"type": "text", "text": user_prompt},
                *[media_ref_to_content_block(ref) for ref in images],
            ]
        conversation = [
            SystemMessage(self._system_prompt),
            HumanMessage(user_content),
        ]
        total_turns = (
            min(len(message.responses), self.include_history)
            if self.include_history >= 0
            else len(message.responses)
        )
        responses = message.responses[-total_turns:]
        for response in responses:
            if response[0] == "user":
                conversation.append(HumanMessage(response[1]))
            elif response[0] == "tool":
                conversation.append(ToolMessage(response[1]))
            elif response[0] == "system":
                # this shouldn't happen but just in case
                conversation.append(SystemMessage(response[1]))
            else:
                conversation.append(AIMessage(response[1]))
        return conversation

    def _generate_response(
        self, conversation: List[BaseMessage], **kwargs
    ) -> Tuple[List[BaseMessage], AIMessage, bool, TokenUsage]:
        response = self._chain.invoke(conversation, **kwargs)
        if len(response.content) > 0 and isinstance(response.content, str):
            response.content = utils.remove_thinking(str(response.content))
        conversation.append(response)
        usage = (
            token_from_response(response.usage_metadata)
            if response.usage_metadata is not None
            else TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        )
        return (
            conversation,
            response,
            len(response.tool_calls) > 0,
            usage,
        )

    def _process_tools(
        self,
        conversation: List[BaseMessage],
        response: AIMessage,
    ) -> List[BaseMessage]:
        for tool_call in response.tool_calls:
            if tool_call["name"] not in self._tool_dict:
                res = tool_call["name"] + " does not exist"
                conversation.append(ToolMessage(res, tool_call_id=tool_call["id"]))
            else:
                try:
                    tool_func = self._tool_dict[tool_call["name"]]
                    if isinstance(tool_func, BaseTool):
                        tool_response = tool_func.invoke(tool_call["args"])
                    elif isinstance(tool_func, Callable):
                        tool_response = tool_func(**tool_call["args"])
                    else:
                        raise ValueError(
                            f"Tool {tool_call['name']} is not a BaseTool or Callable"
                        )
                    conversation.append(
                        ToolMessage(str(tool_response), tool_call_id=tool_call["id"])
                    )
                except Exception as e:
                    res = (
                        f"Encounter error while executing tool {tool_call['name']}. {e}"
                    )
                    conversation.append(ToolMessage(res, tool_call_id=tool_call["id"]))
        return conversation

    def _append_responses(
        self, message: AgentMessage, conversation: List[BaseMessage]
    ) -> AgentMessage:
        start_index = (
            min(len(message.responses), self.include_history) + 2
            if self.store_immediate_steps
            else len(conversation) - 1
        )  # 2 is system message and user query
        end_index = len(conversation)
        name_map = {
            "ai": self.name,
            "human": "user",
            "tool": "tool",
            "system": "system",
        }
        for i in range(start_index, end_index):
            if (
                isinstance(conversation[i].content, str)
                and len(conversation[i].content) > 0
            ):
                message.responses.append(
                    (name_map[conversation[i].type], conversation[i].content)
                )
            # TODO: handle other modals later
        return message

    def bind_tools(
        self, tools: Sequence[Callable | BaseTool], tool_choice: str | None = None
    ) -> None:
        """
        Bind tools to the model.

        Args:
            tools (Sequence[Callable | BaseTool]): A sequence of tools to bind.
            tool_choice (str | None): Follow langchain tool_choice in the
                [bind_tools](https://reference.langchain.com/python/langchain_core/language_models/?h=bind_tools#langchain_core.language_models.BaseChatModel.bind_tools) method

        Returns:
            None
        """
        if len(tools) > 0:
            model_with_tools = self._model.bind_tools(tools, tool_choice=tool_choice)
            self._tool_choice = tool_choice
            if isinstance(tools[0], BaseTool):
                self._tool_dict = {tool.name: tool for tool in tools}  # type: ignore
            else:
                self._tool_dict = {tool.__name__: tool for tool in tools}
            self._chain = model_with_tools
        else:
            self._chain = self._model

    def update_prompt(self, system_prompt: str, user_prompt_template: str) -> None:
        """
        Update the system prompt and user prompt template of the model.
        this will rebuild the template and the LLM chain

        Args:
            system_prompt (str): The new system prompt.
            user_prompt_template (str): The new user prompt template.
        """
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template
        if len(self._tool_dict) > 0:
            model_with_tools = self._model.bind_tools(
                list(self._tool_dict.values()), tool_choice=self._tool_choice
            )
            self._chain = model_with_tools
        else:
            self._chain = self._model

    @property
    def model(self) -> BaseChatModel:
        return self._model

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def user_prompt_template(self) -> str:
        return self._user_prompt_template

    @property
    def tools(self) -> List[Callable | BaseTool]:
        return list(self._tool_dict.values())

    @model.setter
    def model(self, model: BaseChatModel):
        self._model = model
        self._capabilities_cache = None
        if len(self._tool_dict) > 0:
            model_with_tools = self._model.bind_tools(
                list(self._tool_dict.values()),
                tool_choice=self._tool_choice,
            )
            self._chain = model_with_tools
        else:
            self._chain = self._model
