import time

import pytest

from marginalia.concurrency import run_all_sync
from marginalia.timing import Stopwatch


def slow_double(x):
    time.sleep(0.2)  # pretend this is a 0.2 s call to Claude
    return x * 2


def test_results_keep_the_input_order():
    assert run_all_sync(slow_double, [1, 2, 3], max_concurrent=3) == [2, 4, 6]


def test_tasks_run_at_the_same_time():
    with Stopwatch() as sw:
        run_all_sync(slow_double, [1, 2, 3, 4], max_concurrent=4)

    assert sw.elapsed < 0.5  # one after another, 4 x 0.2 s would take 0.8 s


def test_max_concurrent_limits_how_many_run_at_once():
    with Stopwatch() as sw:
        run_all_sync(slow_double, [1, 2, 3, 4], max_concurrent=2)

    assert 0.4 <= sw.elapsed < 0.7  # two waves of two


def test_empty_inputs_give_empty_results():
    assert run_all_sync(slow_double, []) == []


def test_an_error_in_one_task_is_raised():
    def fail_on_3(x):
        if x == 3:
            raise ValueError("bad 3")
        return x

    with pytest.raises(ValueError, match="bad 3"):
        run_all_sync(fail_on_3, [1, 2, 3])
