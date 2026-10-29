import time

import pytest

from marginalia.timing import Stopwatch, timed, timer


def test_timed_returns_the_result_and_reports_the_time(capsys):
    @timed
    def add(a, b):
        return a + b

    assert add(2, b=3) == 5  # positional and named arguments both reach the function

    report = capsys.readouterr().err  # what was printed to stderr
    assert report.startswith("add: ")
    assert report.strip().endswith("s")


def test_timed_keeps_the_function_name_and_docstring():
    @timed
    def greet():
        """Say hi."""

    assert greet.__name__ == "greet"
    assert greet.__doc__ == "Say hi."


def test_timed_reports_even_when_the_function_fails(capsys):
    @timed
    def boom():
        raise ValueError("bad")

    with pytest.raises(ValueError):
        boom()

    assert capsys.readouterr().err.startswith("boom: ")


def test_stopwatch_measures_the_block():
    with Stopwatch() as sw:
        time.sleep(0.02)

    assert 0.02 <= sw.elapsed < 1


def test_stopwatch_gives_itself_to_as():
    stopwatch = Stopwatch()

    with stopwatch as sw:
        pass

    assert sw is stopwatch


def test_stopwatch_lets_errors_through_and_still_measures():
    with pytest.raises(ValueError):
        with Stopwatch() as sw:
            raise ValueError("bad")

    assert sw.elapsed >= 0


def test_timer_prints_the_label_and_time(capsys):
    with timer("search"):
        time.sleep(0.01)

    assert capsys.readouterr().err.startswith("search: ")


def test_timer_prints_even_when_the_block_fails(capsys):
    with pytest.raises(ValueError):
        with timer("boom"):
            raise ValueError("bad")

    assert capsys.readouterr().err.startswith("boom: ")

