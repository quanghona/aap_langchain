# aap_langchain

LangChain integration for the Agent Design Pattern framework.

![PyPI - License](https://img.shields.io/pypi/l/aap_langchain)
![PyPI - Downloads](https://img.shields.io/pypi/dm/aap_langchain)
![PyPI - Version](https://img.shields.io/pypi/v/aap_langchain?label=%20)
![Python Version](https://img.shields.io/pypi/pyversions/aap_langchain)

## What is this?

`aap_langchain` provides LangChain-specific implementations of the core agent abstractions defined in [`aap_core`](../core/). It bridges the gap between the framework's generic agent architecture and LangChain's ecosystem of chat models, tools, and retrievers, enabling you to build agentic workflows with minimal boilerplate.

## Quick Start

```bash
pip install aap_langchain
```

```python
from aap_langchain.chain import ChatCausalMultiTurnsChain
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2:latest", temperature=0.7)
chain = ChatCausalMultiTurnsChain(
    model,
    system_prompt="You are a helpful assistant.",
    user_prompt_template="{query}",
)
```

## Features

- **`ChatCausalMultiTurnsChain`**: A LangChain-backed multi-turn chain that extends `BaseCausalMultiTurnsChain` from `aap_core`. Supports system prompts, user templates, tool calling, and conversation history.
- **`RetrieverAdapter`**: Wraps any LangChain `BaseRetriever` to conform to `aap_core`'s `BaseRetriever` interface, enabling seamless RAG integration.
- **Token usage conversion**: Utility to convert LangChain `UsageMetadata` into `aap_core`'s `TokenUsage` type for consistent tracking across agents.

## Installation

### Core dependency

`aap_langchain` depends on `aap_core`. Install both together:

```bash
pip install aap_core aap_langchain
```

### Optional dependencies

```bash
# Weaviate integration
pip install "aap_langchain[weaviate]"
```

### Development setup

```bash
uv sync --group dev
```

## Usage

### Building an agent with a LangChain model

```python
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from aap_core.agent import BaseAgent
from aap_core.types import AgentMessage
from aap_langchain.chain import ChatCausalMultiTurnsChain
from langchain_ollama import ChatOllama

class MyAgent(BaseAgent):
    chain: BaseLLMChain

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        self.state = "running"
        message = self.chain.invoke(message, **kwargs)
        message.execution_result = "success"
        message.origin = self.card.name
        self.state = "idle"
        return message

model = ChatOllama(model="llama3.2:latest", temperature=0.7)
chain = ChatCausalMultiTurnsChain(
    model,
    system_prompt="You are a creative writer.",
    user_prompt_template="{query}",
)

agent = MyAgent(
    chain=chain,
    card=AgentCard(
        name="writer-agent",
        description="A creative writing agent",
        skills=[AgentSkill(id="writer", name="Writer", description="Writes creative content")],
        capabilities=AgentCapabilities(),
        default_input_modes=["text"],
        default_output_modes=["text"],
        url="localhost",
        version="0.1.0",
    ),
)

result = agent.execute(AgentMessage(query="Write a haiku about rain."))
```

### Using tools with the chain

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

chain = ChatCausalMultiTurnsChain(
    model,
    system_prompt="You are a helpful assistant with weather tools.",
    tools=[get_weather],
    tool_choice="auto",
)
```

### RAG with LangChain retrievers

```python
from aap_langchain.retriever import RetrieverAdapter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(["..."], embedding=embeddings)
retriever = vectorstore.as_retriever()

rag_retriever = RetrieverAdapter(retriever=retriever, data_key="context.retrieved_docs")
```

## Architecture

`aap_langchain` is a subpackage of the [Agent Design Pattern](https://github.com/quanghona/agent_design_pattern) monorepo. It sits on top of `aap_core` and provides LangChain-specific implementations:

```
aap_core          ← Core agent abstractions (BaseAgent, BaseLLMChain, BaseRetriever, AgentMessage)
    └── aap_langchain  ← LangChain integrations
            ├── chain.py      ChatCausalMultiTurnsChain
            ├── retriever.py  RetrieverAdapter
            └── utils.py      token_from_response
```

| Package | Description |
|---------|-------------|
| `aap_core` | Core orchestration framework with agent, chain, and retriever abstractions |
| `aap_langchain` | LangChain integration (this package) |
| `aap_llamaindex` | LlamaIndex integration |
| `aap_dspy` | DSPy integration |
| `aap_transformers` | Hugging Face Transformers integration |

## Documentation

- Example notebooks: [`example/langchain/`](../../example/langchain/)
  - [Simple Loop](../../example/langchain/simple_loop.ipynb)
  - [Sequential](../../example/langchain/sequential.ipynb)
  - [Coordinator](../../example/langchain/coordinator.ipynb)
  - [Self-Reflection](../../example/langchain/self_reflection.ipynb)
  - [Debate](../../example/langchain/debate.ipynb)
  - [Voting](../../example/langchain/voting.ipynb)
  - [Parallel](../../example/langchain/parallel.ipynb)
  - [Cross-Reflection](../../example/langchain/cross_reflection.ipynb)
  - [Iterative Refinement](../../example/langchain/iterative_refinement.ipynb)
  - [Retriever](../../example/langchain/retriever.ipynb)

## Contributing

See the [project contributing guide](../../.github/instructions/project-common-core.instructions.md) for details on development setup, code style, and testing.

## License

This project is licensed under the MIT License — see the [LICENSE](../../LICENSE) file for details.
