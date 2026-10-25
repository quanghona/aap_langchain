"""Contract tests for the TOON prettier (aap_core 0.3.1) through the LangChain adapter.

The TOON encoder itself lives in ``aap_core``: ``DataFrameRetriever.from_pandas``
and ``DataFrameRetriever.from_dict`` accept ``prettier="toon"`` and encode through
``toon_format`` (declared by aap_core as ``toon-format>=1.0.0``). What this module
pins down is the boundary ``aap_langchain`` owns, namely that a TOON payload
survives the trip

    retriever -> AgentMessage.context -> ChatCausalMultiTurnsChain prompt

verbatim. The load-bearing risk is TOON's tabular header, which is brace-bearing
(``"[2]{sku,qty}:"``), meeting ``str.format``-based template interpolation in
``_prepare_conversation``. Values are not re-scanned by ``str.format``, so the
payload must arrive byte-for-byte; these tests fail if that ever stops holding.

Offline by design: the chat model is a local stub that records what it was handed.
No network, no real API calls.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest
import toon_format
from aap_core.retriever import DataFrameRetriever
from aap_core.types import AgentMessage
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.retrievers import BaseRetriever as LangChainBaseRetriever
from pydantic import Field, PrivateAttr

from aap_langchain.chain import ChatCausalMultiTurnsChain
from aap_langchain.retriever import RetrieverAdapter

# Minimum aap_core these contracts require: 0.3.1 routed prettier="toon" to
# toon_format, 0.3.2 made format_kwargs() join list-valued context into text.
TOON_MIN_AAP_CORE = (0, 3, 2)


class RecordingChatModel(BaseChatModel):
    """Chat model stub: records every conversation it is handed, returns fixed text."""

    model_name: str = Field(default="toon-fake")
    reply: str = Field(default="the answer")
    _seen: List[List[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording-fake"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs,
    ) -> ChatResult:
        self._seen.append(list(messages))
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.reply))]
        )

    @property
    def seen(self) -> List[List[BaseMessage]]:
        return self._seen


class SimpleInMemoryRetriever(LangChainBaseRetriever):
    """LangChain retriever over a fixed document list; no embedding or network."""

    documents: List[Document] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self.documents


def make_chain(
    model: Optional[BaseChatModel] = None,
    user_prompt_template: str = "{query}\n{context_data}",
    system_prompt: str = "You are helpful.",
) -> ChatCausalMultiTurnsChain:
    return ChatCausalMultiTurnsChain(
        model=model if model is not None else RecordingChatModel(),
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
    )


def toon_dataframe(records: List[Dict[str, Any]], data_key: str = "context.data"):
    """A DataFrameRetriever whose payload is TOON-encoded, via the public factory."""
    return DataFrameRetriever.from_pandas(
        pd.DataFrame(records), data_key=data_key, prettier="toon"
    )


def human_content_of(chain: ChatCausalMultiTurnsChain, message: AgentMessage) -> str:
    """Rendered user prompt: conversation[0] is the system message."""
    conversation = chain._prepare_conversation(message)
    assert isinstance(conversation[1], HumanMessage)
    return conversation[1].content


def resolved_aap_core_version() -> tuple:
    """Installed aap_core as a comparable tuple, tolerant of suffixes."""
    raw = version("aap_core")
    numbers: List[int] = []
    for chunk in raw.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        numbers.append(int(digits) if digits else 0)
    return tuple(numbers)


class TestResolvedCoreSupportsToon:
    """The upgrade claim: the aap_core this repo resolves actually enables TOON."""

    def test_toon_format_distribution_is_importable(self):
        """toon_format is importable and round-trips, so the core branch can run."""
        assert toon_format.encode({"a": 1}) == "a: 1"
        assert toon_format.decode("a: 1") == {"a": 1}

    def test_aap_core_version_meets_toon_requirement(self):
        """Resolved aap_core is >= 0.3.1, the release that enabled toon_format."""
        try:
            resolved = resolved_aap_core_version()
        except PackageNotFoundError as exc:  # pragma: no cover - env misconfig
            pytest.fail(f"aap_core is not installed in this environment: {exc}")
        assert resolved >= TOON_MIN_AAP_CORE, (
            f"aap_core {resolved} resolved from the environment does not enable "
            f"toon_format; aap_core>={'.'.join(map(str, TOON_MIN_AAP_CORE))} is required. "
            "Bump the dependency floor in pyproject.toml and re-run uv sync."
        )

    def test_from_pandas_routes_toon_to_toon_format(self):
        """prettier="toon" encodes records losslessly instead of falling to tabulate."""
        records = [{"sku": "A-1", "qty": 2}, {"sku": "B-2", "qty": 7}]
        retriever = toon_dataframe(records)
        message = retriever.retrieve(AgentMessage(query="q"))

        payload = message.context["data"]
        assert isinstance(payload, str)
        assert toon_format.decode(payload) == records

    def test_from_dict_routes_toon_to_toon_format(self):
        """prettier="toon" on a dict encodes it losslessly."""
        data = {"warehouse": "north", "capacity": 128}
        retriever = DataFrameRetriever.from_dict(data, prettier="toon")
        message = retriever.retrieve(AgentMessage(query="q"))

        assert toon_format.decode(message.context["data"]) == data

    def test_toon_keyword_is_case_sensitive_and_degrades_silently(self):
        """Only the exact lowercase "toon" routes to toon_format.

        A misspelled prettier is passed to tabulate as a tablefmt, which does not
        reject unknown formats. This test documents the trap: a typo yields a
        plausible-looking string that is NOT TOON, with no exception raised.
        """
        records = [{"sku": "A-1", "qty": 2}]
        typo = DataFrameRetriever.from_pandas(pd.DataFrame(records), prettier="TOON")
        message = typo.retrieve(AgentMessage(query="q"))
        payload = message.context["data"]

        assert "{" not in payload, (
            "prettier='TOON' unexpectedly produced TOON output; the routing "
            "condition in aap_core changed and this canary needs revisiting"
        )
        with pytest.raises(Exception):
            toon_format.decode(payload)


class TestToonPayloadIntoLangChainPrompt:
    """TOON context must reach the LangChain conversation verbatim."""

    def test_dataframe_toon_payload_reaches_human_message(self):
        records = [{"sku": "A-1", "qty": 2}, {"sku": "B-2", "qty": 7}]
        message = toon_dataframe(records).retrieve(AgentMessage(query="total qty?"))
        chain = make_chain()

        prompt = human_content_of(chain, message)

        assert prompt == "total qty?\n" + toon_format.encode(records)

    def test_toon_tabular_header_braces_survive_str_format(self):
        """TOON's "{cols}" header must not be consumed as a format placeholder."""
        records = [{"sku": "A-1", "qty": 2}, {"sku": "B-2", "qty": 7}]
        payload = toon_format.encode(records)
        assert "{" in payload, (
            "TOON no longer emits brace-bearing headers; this regression "
            "canary no longer exercises the risk it was written for"
        )

        message = AgentMessage(query="q", context={"data": payload})
        prompt = human_content_of(make_chain(), message)

        assert payload in prompt
        assert prompt.count(payload) == 1, "payload must appear exactly once"

    def test_from_dict_toon_payload_reaches_human_message(self):
        data = {"a": 1, "b": [1, 2, 3], "c": {"x": "y"}}
        retriever = DataFrameRetriever.from_dict(data, prettier="toon")
        message = retriever.retrieve(AgentMessage(query="inventory"))
        prompt = human_content_of(make_chain(), message)

        assert prompt == "inventory\n" + toon_format.encode(data)
        assert toon_format.decode(prompt.split("\n", 1)[1]) == data

    def test_toon_payload_survives_message_flattening(self):
        """to_dict()/format_kwargs() keep the TOON string intact under context_data."""
        records = [{"sku": "A-1", "qty": 2}]
        payload = toon_format.encode(records)
        message = AgentMessage(query="q", context={"data": payload})

        kwargs = message.format_kwargs()

        assert kwargs["context_data"] == payload
        assert message.to_dict()["context_data"] == payload

    def test_system_message_is_unaffected_by_toon_context(self):
        payload = toon_format.encode([{"sku": "A-1", "qty": 2}])
        message = AgentMessage(query="q", context={"data": payload})
        conversation = make_chain(
            system_prompt="Rules: answer from {data} only"
        )._prepare_conversation(message)

        # The system prompt is not interpolated, so its literal braces stay put.
        assert conversation[0].content == "Rules: answer from {data} only"
        assert payload in conversation[1].content

    @pytest.mark.parametrize(
        "records",
        [
            [{"sku": "A-1", "qty": 2}],
            [
                {"sku": "A-1", "qty": 2},
                {"sku": "B-2", "qty": 7},
                {"sku": "C-3", "qty": 0},
            ],
            [{"name": "café", "note": "你好"}],
            [{"id": 1, "meta": {"x": 1}}, {"id": 2, "meta": {"x": 2}}],
            [{"a": 1, "b": "line1\nline2"}],
        ],
        ids=["single-row", "multi-row", "unicode", "nested-column", "multiline-value"],
    )
    def test_various_tables_are_delivered_verbatim(self, records):
        payload = toon_format.encode(records)
        message = AgentMessage(query="q", context={"data": payload})

        prompt = human_content_of(make_chain(), message)

        assert prompt == "q\n" + payload
        assert toon_format.decode(prompt.split("\n", 1)[1]) == records

    def test_empty_dataframe_encodes_to_empty_table_token(self):
        """Edge case: zero rows must not blow up the prompt path."""
        retriever = DataFrameRetriever.from_pandas(
            pd.DataFrame([{"sku": "A-1", "qty": 2}]).iloc[0:0], prettier="toon"
        )
        message = retriever.retrieve(AgentMessage(query="q"))

        assert message.context["data"] == "[]"
        assert human_content_of(make_chain(), message) == "q\n[]"

    def test_nan_becomes_null_and_stays_prompt_safe(self):
        """Known lossiness: NaN is encoded as TOON null, decoded as None."""
        records = [{"a": 1.0, "b": float("nan")}]
        message = toon_dataframe(records).retrieve(AgentMessage(query="q"))
        payload = message.context["data"]

        assert "null" in payload
        assert toon_format.decode(payload) == [{"a": 1.0, "b": None}]
        assert payload in human_content_of(make_chain(), message)


class TestRetrieverAdapterWithToon:
    """The LangChain-side retriever must not mangle TOON payloads."""

    def test_adapter_stores_toon_encoded_page_content_verbatim(self):
        records = [{"sku": "A-1", "qty": 2}, {"sku": "B-2", "qty": 7}]
        payload = toon_format.encode(records)
        adapter = RetrieverAdapter(
            retriever=SimpleInMemoryRetriever(
                documents=[Document(page_content=payload)]
            )
        )

        result = adapter.retrieve(AgentMessage(query="stock"))

        assert result.context["data"] == [payload]
        assert toon_format.decode(result.context["data"][0]) == records

    def test_adapter_keeps_multiple_toon_documents_separate(self):
        """One TOON table per document: the adapter must not join or re-encode."""
        tables = [[{"id": 1}], [{"id": 2}, {"id": 3}]]
        payloads = [toon_format.encode(t) for t in tables]
        adapter = RetrieverAdapter(
            retriever=SimpleInMemoryRetriever(
                documents=[Document(page_content=p) for p in payloads]
            )
        )

        result = adapter.retrieve(AgentMessage(query="q"))

        assert result.context["data"] == payloads
        assert [toon_format.decode(p) for p in result.context["data"]] == tables

    def test_adapter_custom_data_key_holds_toon_payload(self):
        payload = toon_format.encode([{"sku": "A-1"}])
        adapter = RetrieverAdapter(
            retriever=SimpleInMemoryRetriever(
                documents=[Document(page_content=payload)]
            ),
            data_key="context.stock",
        )

        result = adapter.retrieve(AgentMessage(query="q"))

        assert result.context == {"stock": [payload]}
        assert toon_format.decode(result.context["stock"][0]) == [{"sku": "A-1"}]

    def test_adapter_toon_payload_is_decodable_after_prompt_interpolation(self):
        """A TOON table retrieved via a LangChain retriever must stay decodable."""
        payload = toon_format.encode([{"sku": "A-1"}])
        adapter = RetrieverAdapter(
            retriever=SimpleInMemoryRetriever(
                documents=[Document(page_content=payload)]
            ),
            data_key="context.stock",
        )
        result = adapter.retrieve(AgentMessage(query="q"))

        prompt = human_content_of(
            make_chain(user_prompt_template="{query}\n{context_stock}"), result
        )
        toon_format.decode(prompt.split("\n", 1)[1])

    def test_adapter_with_toon_dataframe_post_process_merges_contexts(self):
        """Composition: LangChain docs plus a TOON table in one message context."""
        doc_payload = toon_format.encode([{"chunk": "retrieved text"}])
        table_records = [{"sku": "A-1", "qty": 2}]
        adapter = RetrieverAdapter(
            retriever=SimpleInMemoryRetriever(
                documents=[Document(page_content=doc_payload)]
            ),
            post_process=toon_dataframe(table_records, data_key="context.table"),
        )

        result = adapter(AgentMessage(query="q"))

        assert result.context["data"] == [doc_payload]
        assert toon_format.decode(result.context["table"]) == table_records

    def test_adapter_rejects_non_prefixed_data_key_for_toon_context(self):
        with pytest.raises(ValueError, match="data_key must start with 'context.'"):
            RetrieverAdapter(
                retriever=SimpleInMemoryRetriever(documents=[]),
                data_key="toon_data",
            )


class TestToonEndToEnd:
    """Retriever -> chain -> AgentMessage, with a stubbed model."""

    def test_retrieve_then_invoke_delivers_toon_to_the_model(self):
        records = [{"sku": "A-1", "qty": 2}, {"sku": "B-2", "qty": 7}]
        payload = toon_format.encode(records)
        model = RecordingChatModel(reply="9 units")
        chain = make_chain(model=model)
        message = toon_dataframe(records)(AgentMessage(query="total qty?"))

        result = chain.invoke(message)

        assert model.seen[0][1].content == f"total qty?\n{payload}"
        assert result.responses == [("chain", "9 units")]
        assert result.execution_result == "success"
        assert result.context["data"] == payload, "context must survive the chain"

    def test_invoke_round_trips_toon_context_back_into_agent_message(self):
        """Round-trip: TOON in context -> prompt -> out again -> decodes."""
        records = [{"sku": "A-1", "qty": 2}]
        payload = toon_format.encode(records)
        chain = make_chain(model=RecordingChatModel())
        message = chain.invoke(AgentMessage(query="q", context={"data": payload}))

        assert toon_format.decode(message.context["data"]) == records
        assert toon_format.decode(message.to_dict()["context_data"]) == records

    def test_toon_context_is_not_duplicated_by_history(self):
        """TOON lands in the current user turn only, never in replayed history."""
        payload = toon_format.encode([{"sku": "A-1", "qty": 2}])
        model = RecordingChatModel()
        chain = make_chain(model=model)
        chain.include_history = 2
        message = AgentMessage(
            query="q",
            context={"data": payload},
            responses=[("user", "earlier question"), ("ai", "earlier answer")],
        )

        conversation = chain._prepare_conversation(message)

        assert conversation[1].content == f"q\n{payload}"
        assert [m.type for m in conversation] == ["system", "human", "human", "ai"]
        assert all(payload not in str(m.content) for m in conversation[2:])
