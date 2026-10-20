"""One LLM call: answer a question using only the chunks that search found.

Run from the project root (costs real money: one Claude Haiku 4.5 call per question):
    HF_HUB_OFFLINE=1 uv run --env-file .env python -m marginalia.answer "что означает RSI выше 70?"
"""

import sys
from pathlib import Path

import anthropic
from anthropic.types import Usage
from pydantic import BaseModel, Field

from marginalia.index import PATH_SEP, Hit

MODEL = "claude-haiku-4-5"
PRICE_PER_MTOK = {"input": 1.00, "output": 5.00}  # USD, Claude Haiku 4.5

SYSTEM = """You answer questions about trading using ONLY the book excerpts in the user message.
- Every claim must come from the excerpts. Do not add outside knowledge.
- If the excerpts do not answer the question, set not_in_books to true and say so in one sentence.
- Answer in the language of the question. Keep trading terms as they are in the book.
- In sources, list the id in square brackets of every excerpt you used, for example murphy-ta:318."""


class Answer(BaseModel):
    answer: str = Field(description="The answer, based only on the excerpts")
    sources: list[str] = Field(description="Ids of the excerpts used, e.g. murphy-ta:318")
    not_in_books: bool = Field(description="True if the excerpts do not contain the answer")


def format_excerpts(hits: list[Hit]) -> str:
    """Each excerpt gets its id and heading path, so the model can cite it."""
    return "\n\n---\n\n".join(f"[{h.chunk_id}] {PATH_SEP.join(h.heading_path)}\n{h.text}" for h in hits)


def answer_question(client: anthropic.Anthropic, question: str, hits: list[Hit]) -> tuple[Answer, Usage]:
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"<excerpts>\n{format_excerpts(hits)}\n</excerpts>\n\nQuestion: {question}",
            }
        ],
        output_format=Answer,  # the SDK makes the model return exactly this shape and validates it
    )
    if response.parsed_output is None:
        # e.g. stop_reason "max_tokens" (cut off) or "refusal": there is no valid JSON to read
        raise RuntimeError(f"no structured answer, stop_reason={response.stop_reason}")
    return response.parsed_output, response.usage


def cost_usd(usage: Usage) -> float:
    return (
        usage.input_tokens * PRICE_PER_MTOK["input"] + usage.output_tokens * PRICE_PER_MTOK["output"]
    ) / 1_000_000


if __name__ == "__main__":
    from marginalia.embeddings import E5Embedder
    from marginalia.index import open_collection, search

    question = sys.argv[1]
    collection = open_collection(Path("data/chroma"))
    hits = search(collection, E5Embedder(), question, k=5)

    answer, usage = answer_question(anthropic.Anthropic(), question, hits)

    print(f"\n{answer.answer}\n")
    print(f"not_in_books: {answer.not_in_books}")
    paths = {h.chunk_id: PATH_SEP.join(h.heading_path) for h in hits}
    for source in answer.sources:
        print(f"  [{source}] {paths.get(source, '⚠ not among the found chunks')}")
    print(f"\ntokens: {usage.input_tokens} in / {usage.output_tokens} out  ≈ ${cost_usd(usage):.4f}")
