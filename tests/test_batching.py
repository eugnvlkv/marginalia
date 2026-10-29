import pytest

from marginalia.batching import batched


def test_splits_into_batches_of_size():
    assert list(batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_exact_multiple_has_no_empty_tail():
    assert list(batched([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]


def test_empty_input_gives_nothing():
    assert list(batched([], 3)) == []


def test_is_lazy():
    pulled = []  # every number the source has produced so far

    def numbers():
        for n in range(1_000_000):
            pulled.append(n)
            yield n

    first = next(batched(numbers(), 3))  # ask for ONE batch

    assert first == [0, 1, 2]
    assert pulled == [0, 1, 2]  # nothing beyond the first batch was produced: no million-item list


def test_bad_size_raises_on_first_use():
    # A generator function runs nothing until it is iterated, so the check fires inside list(...).
    with pytest.raises(ValueError):
        list(batched([1, 2], 0))
