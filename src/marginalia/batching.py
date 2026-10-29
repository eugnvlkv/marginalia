"""batched: split any iterable into lists of at most `size` items — lazily, one batch at a time.

Exercise: fill in the four `...` below, then run
    uv run pytest tests/test_batching.py -v
"""

from collections.abc import Iterable, Iterator


def batched(items: Iterable, size: int) -> Iterator[list]:
    if size < 1:
        raise ValueError("size must be at least 1")

    batch = []  # the current batch: the same "box" as in split_text
    for item in items:
        batch.append(item)
        if len(batch) == size:  # (2) is the batch full?
            yield batch  # hand the batch out; the function PAUSES here until the next batch is asked for
            batch = []  # (3) start a new, empty batch
    if len(batch) > 0:  # (4) anything left at the end?
        yield batch
