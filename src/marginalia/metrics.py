"""Retrieval metrics, written by hand: recall@k and MRR.

    retrieved — the sections search returned, in rank order (1st, 2nd, ...). The same section can
                appear twice: search found two chunks of it.
    relevant  — the sections that answer the question (from the golden set).

Exercise: fill in the six `...` below, then run
    uv run pytest tests/test_metrics.py -v
"""


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """What share of the right sections is among the first k results."""
    if not relevant:
        raise ValueError("recall is undefined without relevant items")  # unanswerable questions are scored differently

    # (1) Срез: первые k элементов. xs[:k] — «с начала до k, не включая k».
    #     JS: retrieved.slice(0, k). Если k больше длины списка — не ошибка, вернётся весь список.
    top_k = retrieved[:k]
    # (2) set(top_k) — множество из списка: повторы исчезают (два куска одного раздела — одна находка).
    #     & — пересечение: только то, что есть в обоих множествах.
    #     {"Elliott Wave", "RSI"} & {"RSI", "Глоссарий"} → {"RSI"}.  JS: new Set(topK).intersection(relevant)
    found = set(top_k) & relevant
    # (3) Доля найденных: сколько правильных нашли ÷ сколько правильных всего (1 из 2 → 0.5).
    return len(found) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / place of the first right section (places start at 1), or 0.0 if none was found."""
    # enumerate(..., start=1) даёт пары (место, раздел): (1, первый), (2, второй), ...
    # JS: for (const [i, section] of retrieved.entries()) { const place = i + 1; ... }
    for place, section in enumerate(retrieved, start=1):
        # (4) x in s — есть ли элемент во множестве. JS: relevant.has(section)
        if section in relevant:
            # (5) return внутри цикла сразу выходит из функции, поэтому RR видит только ПЕРВЫЙ правильный.
            #     Так и задумано. Остальные правильные учитывают recall и MAP.
            return 1/place
    return 0.0  # цикл дошёл до конца — правильного раздела в выдаче нет


def mean_reciprocal_rank(results: list[tuple[list[str], set[str]]]) -> float:
    """MRR over many questions: the average of their reciprocal ranks. Each result is (retrieved, relevant)."""
    if not results:
        raise ValueError("MRR is undefined without questions")
    kek = 0
    # (6) for a, b in pairs раскладывает каждую пару (выдача, правильные) на два имени.
    #     JS: for (const [retrieved, relevant] of results)
    for retrieved, relevant in results:
        kek += reciprocal_rank(retrieved, relevant)
    return kek / len(results)  # сумма RR ÷ число вопросов = среднее