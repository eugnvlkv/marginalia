"""The golden set: load evals/golden.toml, check it against the index, help find and review sections.

    uv run python -m marginalia.golden sections <word>   # exact section paths to paste into golden.toml
    uv run python -m marginalia.golden check [file]      # is the file valid and do all paths exist?
    uv run python -m marginalia.golden show <id> [file]  # one question + the full text of its sections, to review
    uv run python -m marginalia.golden promote           # move `reviewed = true` blocks from candidates to golden
"""

import re
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Literal

import chromadb
from pydantic import BaseModel, ConfigDict, Field

from marginalia.index import PATH_SEP

GOLDEN_PATH = Path("evals/golden.toml")
CANDIDATES_PATH = Path("evals/candidates.toml")


class Relevant(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo in a field name is an error, not silently ignored

    book: str
    path: str  # "Chapter › Section"; also covers the section's subsections


class GoldenQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    # partial: part of the question is in the books, part is not (e.g. "how does X work, and what is it on BTC now?")
    kind: Literal["factual", "multi", "partial", "unanswerable"]
    question: str = Field(min_length=1)
    relevant: list[Relevant] = []
    expected: str = ""
    reviewed: bool = False  # kind, relevant and expected were checked against the book text (by a human or an independent review)


class GoldenSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[GoldenQuestion] = []  # a file with only comments is an empty set, not an error


def load_golden(path: Path = GOLDEN_PATH) -> GoldenSet:
    with path.open("rb") as file:  # tomllib reads bytes
        return GoldenSet.model_validate(tomllib.load(file))


def all_sections(collection: chromadb.Collection) -> set[tuple[str, str]]:
    """Every (book, section path) in the index."""
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    return {(meta["book_id"], meta["heading_path"]) for meta in metadatas}


def covers(section_path: str, chunk_path: str) -> bool:
    """A section covers itself and everything nested under it."""
    return chunk_path == section_path or chunk_path.startswith(section_path + PATH_SEP)


def check_golden(golden: GoldenSet, collection: chromadb.Collection) -> list[str]:
    """Problems that would make the numbers lie. An empty list means the golden set is usable."""
    problems = []
    sections = all_sections(collection)

    for question_id, count in Counter(q.id for q in golden.questions).items():
        if count > 1:
            problems.append(f"{question_id}: id is used {count} times")

    for q in golden.questions:
        if q.kind == "unanswerable" and q.relevant:
            problems.append(f"{q.id}: unanswerable but has relevant sections")
        if q.kind != "unanswerable" and not q.relevant:
            problems.append(f"{q.id}: {q.kind} needs at least one relevant section")
        if q.kind == "multi" and len(q.relevant) < 2:
            problems.append(f"{q.id}: multi needs at least two relevant sections")
        if q.kind != "unanswerable" and not q.expected.strip():
            problems.append(f"{q.id}: expected is empty — the judge has nothing to compare with")
        for r in q.relevant:
            if not any(book == r.book and covers(r.path, path) for book, path in sections):
                problems.append(f"{q.id}: no section {r.path!r} in book {r.book!r}")

    return problems


def section_text(collection: chromadb.Collection, book_id: str, path: str) -> str:
    """The text of a section and all its subsections, in book order, with a title line per (sub)section."""
    result = collection.get(where={"book_id": book_id}, include=["documents", "metadatas"])
    rows = sorted(
        (meta["position"], meta["heading_path"], text)
        for text, meta in zip(result["documents"], result["metadatas"])
        if covers(path, meta["heading_path"])
    )
    parts, last_path = [], None
    for _, chunk_path, text in rows:
        if chunk_path != last_path:
            parts.append(f"── {chunk_path}")
            last_path = chunk_path
        parts.append(text)
    return "\n\n".join(parts)


QUESTION_START = re.compile(r"^\[\[questions\]\]", flags=re.MULTILINE)


def promote(source: Path, target: Path) -> list[str]:
    """Move every block with `reviewed = true` from source to the end of target, comments included.

    Returns the ids that were moved.
    """
    head, *blocks = QUESTION_START.split(source.read_text(encoding="utf-8"))
    keep, move = [], []
    for block in blocks:
        question = tomllib.loads("[[questions]]" + block)["questions"][0]
        (move if question.get("reviewed") else keep).append(block)
    if move:
        moved_text = "".join("[[questions]]" + block for block in move)
        target.write_text(target.read_text(encoding="utf-8").rstrip() + "\n\n" + moved_text, encoding="utf-8")
        source.write_text(head + "".join("[[questions]]" + block for block in keep), encoding="utf-8")
    return [tomllib.loads("[[questions]]" + block)["questions"][0]["id"] for block in move]


def find_sections(collection: chromadb.Collection, word: str) -> list[tuple[str, str]]:
    """Sections whose path contains the word (any case), sorted."""
    return sorted((book, path) for book, path in all_sections(collection) if word.lower() in path.lower())


if __name__ == "__main__":
    from marginalia.index import open_collection

    collection = open_collection(Path("data/chroma"))
    command = sys.argv[1]

    if command == "sections":
        for book, path in find_sections(collection, sys.argv[2]):
            print(f'    {{ book = "{book}", path = "{path}" }},')
    elif command == "check":
        golden = load_golden(Path(sys.argv[2]) if len(sys.argv) > 2 else GOLDEN_PATH)
        kinds = Counter(q.kind for q in golden.questions)
        print(f"{len(golden.questions)} questions: " + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items())))
        print(f"reviewed: {sum(q.reviewed for q in golden.questions)} of {len(golden.questions)}")
        problems = check_golden(golden, collection)
        for problem in problems:
            print(f"  ✗ {problem}")
        sys.exit(1 if problems else 0)
    elif command == "show":
        source = Path(sys.argv[3]) if len(sys.argv) > 3 else CANDIDATES_PATH
        question = next((q for q in load_golden(source).questions if q.id == sys.argv[2]), None)
        if question is None:
            sys.exit(f"no question {sys.argv[2]!r} in {source}")
        print(f"{question.id}  [{question.kind}]  reviewed={question.reviewed}\n")
        print(f"ВОПРОС: {question.question}\n")
        print(f"EXPECTED:{question.expected.rstrip()}\n")
        for r in question.relevant:
            print(f"══════ {r.book} · {r.path}\n")
            print(section_text(collection, r.book, r.path) + "\n")
    elif command == "promote":
        moved = promote(CANDIDATES_PATH, GOLDEN_PATH)
        print(f"moved {len(moved)} to {GOLDEN_PATH}: {', '.join(moved) or '—'}")
    else:
        sys.exit(f"unknown command: {command} (use sections, check, show or promote)")
