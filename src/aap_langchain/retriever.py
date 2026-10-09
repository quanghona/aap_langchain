from aap_core.retriever import BaseRetriever
from aap_core.types import AgentMessage
from langchain_core.retrievers import BaseRetriever as LangChainBaseRetriever
from pydantic import Field, field_validator


class RetrieverAdapter(BaseRetriever):
    retriever: LangChainBaseRetriever = Field(
        ..., description="The langchain's retriever to use"
    )
    data_key: str = Field(
        default="context.data", description="The key to the data in the message"
    )

    @field_validator("data_key")
    @classmethod
    def check_starts_with_prefix(cls, v: str) -> str:
        if not v.startswith("context."):
            raise ValueError("data_key must start with 'context.'")
        return v

    def retrieve(self, message: AgentMessage, **kwargs) -> AgentMessage:
        docs = self.retriever.invoke(message.query)
        data = []
        # TODO: aware for multimodal data in the future
        for doc in docs:
            data.append(doc.page_content)

        data_key = self.data_key.replace("context.", "")
        if message.context is None:
            message.context = {data_key: data}
        else:
            message.context[data_key] = data
        return message
