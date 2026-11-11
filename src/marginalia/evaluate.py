"""Retrieval evals: ask the real search every golden question and score its top k with recall@k and MRR.

    HF_HUB_OFFLINE=1 uv run python -m marginalia.evaluate [k]    # needs the index in data/chroma; k = 5 by default

Search returns chunks, the golden set names sections. label_hits turns every found chunk into the key of
the right section it lies in, so the functions from metrics.py can compare them. Unanswerable questions
have no right sections and are left out: they are checked on the answer (an honest refusal), in step 4.

Exercise: fill in the three `...` in label_hits, then run
    uv run pytest tests/test_evaluate.py -v
"""

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import chromadb

from marginalia.embeddings import Embedder
from marginalia.golden import GoldenQuestion, GoldenSet, Relevant, covers, load_golden
from marginalia.index import PATH_SEP, Hit, open_collection, search
from marginalia.metrics import mean_reciprocal_rank, recall_at_k


def section_key(book: str, path: str) -> str:
    """One string per section, so that sections of different books never mix up."""
    return f"{book}: {path}"


def label_hits(hits: list[Hit], relevant: list[Relevant]) -> list[str]:
    """For each hit, in search order: the key of the right section it lies in, or its own key if it lies in none."""
    labels = []
    for hit in hits:
        path = PATH_SEP.join(hit.heading_path)  # ("5 Major…", "VOLUME") → "5 Major… › VOLUME"
        # (1) По умолчанию кусок не из правильного раздела: его собственный ключ — section_key(книга, путь).
        label = section_key(hit.book_id, path)
        for section in relevant:
            # (2) Та же книга (section.book и hit.book_id) И правильный раздел покрывает путь куска (covers из golden.py).
            if section.book == hit.book_id and covers(section.path, path):
                # (3) Нашли: кусок засчитывается правильному разделу — ключ этого раздела.
                label = section_key(section.book, section.path)
                break  # свой раздел нашли, остальные можно не смотреть
        labels.append(label)
    return labels


def outermost(relevant: list[Relevant]) -> list[Relevant]:
    """Drop right sections nested in another right section of the same question: the outer one already covers them."""
    return [
        r
        for r in relevant
        if not any(o.book == r.book and o.path != r.path and covers(o.path, r.path) for o in relevant)
    ]


@dataclass(frozen=True)
class Scored:
    """One golden question after search: what came back, as section keys, and which keys are right."""

    question: GoldenQuestion
    labels: list[str]
    relevant: set[str]

    @property
    def missed(self) -> list[str]:
        return sorted(self.relevant - set(self.labels))


def score(question: GoldenQuestion, collection: chromadb.Collection, embedder: Embedder, k: int) -> Scored:
    relevant = outermost(question.relevant)
    hits = search(collection, embedder, question.question, k=k)
    return Scored(
        question=question,
        labels=label_hits(hits, relevant),
        relevant={section_key(r.book, r.path) for r in relevant},
    )


def evaluate(golden: GoldenSet, collection: chromadb.Collection, embedder: Embedder, k: int = 5) -> list[Scored]:
    """Score every answerable question; unanswerable ones have no right sections to find."""
    return [score(q, collection, embedder, k) for q in golden.questions if q.kind != "unanswerable"]


def summarize(results: list[Scored], k: int) -> dict[str, tuple[int, float, float]]:
    """Per kind and for "all": (number of questions, mean recall@k, MRR)."""
    groups: dict[str, list[Scored]] = defaultdict(list)
    for r in results:
        groups[r.question.kind].append(r)
        groups["all"].append(r)
    return {
        kind: (
            len(group),
            sum(recall_at_k(r.labels, r.relevant, k) for r in group) / len(group),
            mean_reciprocal_rank([(r.labels, r.relevant) for r in group]),
        )
        for kind, group in groups.items()
    }


if __name__ == "__main__":
    from marginalia.embeddings import E5Embedder

    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    golden = load_golden()
    results = evaluate(golden, open_collection(Path("data/chroma")), E5Embedder(), k)
    summary = summarize(results, k)

    print(f"k = {k}: {len(results)} questions, {len(golden.questions) - len(results)} unanswerable left out\n")
    print(f"{'kind':<9} {'n':>3}  {f'recall@{k}':>9}  {'MRR':>5}")
    for kind in ("factual", "multi", "partial", "all"):
        if kind in summary:
            n, recall, mrr = summary[kind]
            print(f"{kind:<9} {n:>3}  {recall:>9.2f}  {mrr:>5.2f}")

    misses = [r for r in results if r.missed]
    print(f"\nRight sections not in the top {k} ({len(misses)} questions):")
    for r in misses:
        print(f"  {r.question.id} [{r.question.kind}]  recall {recall_at_k(r.labels, r.relevant, k):.2f}")
        for key in r.missed:
            print(f"      − {key}")
