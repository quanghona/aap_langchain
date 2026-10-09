import pytest
from aap_core.types import AgentMessage
from aap_langchain.retriever import RetrieverAdapter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever as LangChainBaseRetriever


class SimpleInMemoryRetriever(LangChainBaseRetriever):
    """A simple in-memory retriever for testing purposes."""

    documents: list[Document] = []

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return self.documents


class TestRetrieverAdapter:
    """Tests for RetrieverAdapter."""

    def _create_retriever(self, documents: list[Document]) -> RetrieverAdapter:
        lc_retriever = SimpleInMemoryRetriever(documents=documents)
        return RetrieverAdapter(retriever=lc_retriever)

    def test_retrieve_returns_message_with_context(self):
        """Test that retrieve adds retrieved documents to message context."""
        documents = [
            Document(page_content="Document one content"),
            Document(page_content="Document two content"),
        ]
        adapter = self._create_retriever(documents)
        message = AgentMessage(query="test query")

        result = adapter.retrieve(message)

        assert result.context is not None
        assert "data" in result.context
        assert result.context["data"] == [
            "Document one content",
            "Document two content",
        ]

    def test_retrieve_default_data_key(self):
        """Test that default data_key is 'context.data' and stored as 'data'."""
        documents = [
            Document(page_content="Relevant doc"),
        ]
        adapter = self._create_retriever(documents)
        message = AgentMessage(query="test query")

        result = adapter.retrieve(message)

        assert result.context == {"data": ["Relevant doc"]}

    def test_retrieve_custom_data_key(self):
        """Test that custom data_key is correctly stripped of 'context.' prefix."""
        documents = [
            Document(page_content="Custom key doc"),
        ]
        adapter = RetrieverAdapter(
            retriever=SimpleInMemoryRetriever(documents=documents),
            data_key="context.custom_docs",
        )
        message = AgentMessage(query="test query")

        result = adapter.retrieve(message)

        assert result.context == {"custom_docs": ["Custom key doc"]}

    def test_retrieve_merges_with_existing_context(self):
        """Test that retrieve merges with existing message context."""
        documents = [
            Document(page_content="New doc"),
        ]
        adapter = self._create_retriever(documents)
        message = AgentMessage(
            query="test query",
            context={"existing_key": "existing_value"},
        )

        result = adapter.retrieve(message)

        assert result.context == {
            "existing_key": "existing_value",
            "data": ["New doc"],
        }

    def test_retrieve_overwrites_existing_data_key(self):
        """Test that retrieve overwrites existing data under the same key."""
        documents = [
            Document(page_content="Updated doc"),
        ]
        adapter = self._create_retriever(documents)
        message = AgentMessage(
            query="test query",
            context={"data": ["Old doc"]},
        )

        result = adapter.retrieve(message)

        assert result.context["data"] == ["Updated doc"]

    def test_retrieve_empty_documents(self):
        """Test that retrieve handles empty document list gracefully."""
        adapter = self._create_retriever([])
        message = AgentMessage(query="test query")

        result = adapter.retrieve(message)

        assert result.context == {"data": []}

    def test_data_key_validator_rejects_invalid_prefix(self):
        """Test that data_key must start with 'context.'."""
        with pytest.raises(ValueError, match="data_key must start with 'context.'"):
            RetrieverAdapter(
                retriever=SimpleInMemoryRetriever(documents=[]),
                data_key="invalid_key",
            )

    def test_data_key_attribute_assignment(self):
        """Test that data_key attribute can be reassigned and the new value is used."""
        documents = [
            Document(page_content="Assigned key doc"),
        ]
        adapter = self._create_retriever(documents)
        adapter.data_key = "context.assigned_key"

        message = AgentMessage(query="test query")
        result = adapter.retrieve(message)

        assert result.context == {"assigned_key": ["Assigned key doc"]}

    def test_data_key_validator_accepts_valid_prefix(self):
        """Test that data_key with 'context.' prefix is accepted."""
        adapter = RetrieverAdapter(
            retriever=SimpleInMemoryRetriever(documents=[]),
            data_key="context.valid_key",
        )
        assert adapter.data_key == "context.valid_key"

    def test_retrieve_preserves_other_message_fields(self):
        """Test that retrieve preserves original message fields."""
        documents = [
            Document(page_content="Doc content"),
        ]
        adapter = self._create_retriever(documents)
        message = AgentMessage(
            query="original query",
            origin="test_agent",
        )

        result = adapter.retrieve(message)

        assert result.query == "original query"
        assert result.origin == "test_agent"
        assert result.responses == []
