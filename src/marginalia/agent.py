"""Agent: Claude decides when to search, when to read a whole section, and when it can answer.

LangGraph holds the loop as a graph; Claude is called directly through the anthropic SDK.
Run from the project root (costs real money: usually 2-4 Claude Haiku 4.5 calls per question):
    HF_HUB_OFFLINE=1 uv run --env-file .env python -m marginalia.agent "что означает RSI выше 70?"
"""

import json
import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

import anthropic
import chromadb
from anthropic.types import Usage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from marginalia.answer import MODEL, Answer, cost_usd
from marginalia.embeddings import Embedder
from marginalia.index import PATH_SEP, Hit, get_section, search
from marginalia.timing import timed


MAX_MODEL_CALLS = 6  # a safety net: the agent must not loop (and spend) forever

SYSTEM = """You answer questions about trading from a library of trading books. Use the tools to read the books.
- Always write the answer in the language of the question, including when the books do not cover it.
  Keep trading terms as they are in the book.
- The books are in English: call search_docs with English queries, even if the question is in another language.
  If the first search finds nothing useful, try other wording.
- If an excerpt looks relevant but incomplete, call get_section to read the whole section.
- Answer ONLY from what the tools returned. Do not add outside knowledge.
- If the books do not answer the question, set not_in_books to true and say so in one sentence.
- In sources, list the chunk ids you used, for example murphy-ta:318."""

TOOLS = [
    {
        "name": "search_docs",
        "description": (
            "Semantic search over the trading books. Returns the 5 closest excerpts, "
            "each with its chunk id, book_id and heading_path."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look for, in English"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_section",
        "description": (
            "Read a whole book section: all excerpts that share one heading path, in book order. "
            "Take book_id and heading_path from a search_docs result."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "book_id": {"type": "string"},
                "heading_path": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Titles from the chapter down to the section, exactly as in the search result",
                },
            },
            "required": ["book_id", "heading_path"],
            "additionalProperties": False,
        },
    },
]


class State(TypedDict):
    # operator.add is the "reducer": what a node returns under this key is APPENDED, not replaced.
    messages: Annotated[list, operator.add]
    usage: Annotated[list, operator.add]
    answer: Answer | None


class ToolError(Exception):
    """A tool could not do what the model asked; the model gets the message and can try again."""


def format_hits(hits: list[Hit]) -> str:
    """Excerpts with everything the model needs to cite them and to call get_section."""
    if not hits:
        return "No excerpts found."
    return "\n\n---\n\n".join(
        f"[{h.chunk_id}] book_id={h.book_id} heading_path={json.dumps(list(h.heading_path), ensure_ascii=False)}\n"
        f"{h.text}"
        for h in hits
    )


def to_param(block) -> dict:
    """A response block as a plain dict we can send back in the next request."""
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if block.type == "text":
        return {"type": "text", "text": block.text}
    # Haiku without thinking returns only these two. A thinking model would need its
    # thinking blocks sent back unchanged, so fail loudly instead of dropping them.
    raise ValueError(f"unexpected block type: {block.type}")


def build_agent(client: anthropic.Anthropic, collection: chromadb.Collection, embedder: Embedder):
    """The graph: call_model → (tool_use? run_tools → call_model : END)."""

    @timed
    def run_tool(name: str, args: dict) -> str:
        if name == "search_docs":
            return format_hits(search(collection, embedder, args["query"], k=5))
        if name == "get_section":
            section = get_section(collection, args["book_id"], tuple(args["heading_path"]))
            if section is None:
                raise ToolError(f"no section {args['heading_path']} in book {args['book_id']}")
            return f"[{', '.join(section.chunk_ids)}] {PATH_SEP.join(section.heading_path)}\n{section.text}"
        raise ToolError(f"unknown tool: {name}")

    @timed
    def call_model(state: State) -> dict:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            tools=TOOLS,
            output_format=Answer,  # used for the final answer; tool calls come back as tool_use blocks
            messages=state["messages"],
        )
        update = {
            "messages": [{"role": "assistant", "content": [to_param(b) for b in response.content]}],
            "usage": [response.usage],
        }
        if response.stop_reason != "tool_use":
            if response.parsed_output is None:
                raise RuntimeError(f"no structured answer, stop_reason={response.stop_reason}")
            update["answer"] = response.parsed_output
        return update

    def run_tools(state: State) -> dict:
        # The last message is the model's turn; run every tool it asked for, answer all in ONE user message.
        results = []
        for block in state["messages"][-1]["content"]:
            if block["type"] != "tool_use":
                continue
            try:
                content, is_error = run_tool(block["name"], block["input"]), False
            except ToolError as error:
                content, is_error = f"Error: {error}", True
            results.append(
                {"type": "tool_result", "tool_use_id": block["id"], "content": content, "is_error": is_error}
            )
        return {"messages": [{"role": "user", "content": results}]}

    def route(state: State) -> str:
        return END if state["answer"] is not None else "run_tools"

    graph = StateGraph(State)
    graph.add_node("call_model", call_model)
    graph.add_node("run_tools", run_tools)
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges("call_model", route, ["run_tools", END])
    graph.add_edge("run_tools", "call_model")
    return graph.compile()


def run_agent(agent, question: str, max_model_calls: int = MAX_MODEL_CALLS) -> State:
    start: State = {"messages": [{"role": "user", "content": question}], "usage": [], "answer": None}
    try:
        # Each model call and each tool round is one graph step, so N calls take at most 2N - 1 steps.
        return agent.invoke(start, config={"recursion_limit": 2 * max_model_calls - 1})
    except GraphRecursionError:
        raise RuntimeError(f"the agent did not answer within {max_model_calls} model calls") from None


def tool_calls(state: State) -> list[tuple[str, dict]]:
    """(tool name, input) for every tool call the agent made, in order."""
    return [
        (block["name"], block["input"])
        for message in state["messages"]
        if message["role"] == "assistant"
        for block in message["content"]
        if block["type"] == "tool_use"
    ]


if __name__ == "__main__":
    from marginalia.embeddings import E5Embedder
    from marginalia.index import open_collection

    agent = build_agent(anthropic.Anthropic(), open_collection(Path("data/chroma")), E5Embedder())
    state = run_agent(agent, sys.argv[1])

    for name, args in tool_calls(state):
        print(f"→ {name} {json.dumps(args, ensure_ascii=False)}")
    answer = state["answer"]
    print(f"\n{answer.answer}\n")
    print(f"not_in_books: {answer.not_in_books}")
    print(f"sources: {', '.join(answer.sources) or '—'}")

    usage: list[Usage] = state["usage"]
    tokens_in = sum(u.input_tokens for u in usage)
    tokens_out = sum(u.output_tokens for u in usage)
    total = sum(cost_usd(u) for u in usage)
    print(f"\nmodel calls: {len(usage)}, tokens: {tokens_in} in / {tokens_out} out  ≈ ${total:.4f}")
