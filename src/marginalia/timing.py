"""Timing helpers: @timed (decorator), Stopwatch and timer (context managers).

Exercises: @timed is done. Fill in the `...` in Stopwatch and timer below, then run
    uv run pytest tests/test_timing.py -v
"""

import functools
import sys
import time
from contextlib import contextmanager


def timed(func):
    @functools.wraps(func)  # copy the name and docstring of func onto the wrapper
    def wrapper(*args, **kwargs):  # accept ANY arguments: positional in args, named in kwargs
        start = time.perf_counter()
        try:
            return func(*args, **kwargs) # (2) call func with the same arguments and return its result
        finally:  # runs both on success and on error
            elapsed = time.perf_counter() - start   # (3) how many seconds passed since start
            print(f"{func.__name__}: {elapsed:.2f}s", file=sys.stderr)

    return wrapper


class Stopwatch:
    """Measure a block of code: `with Stopwatch() as sw: ...`, then read sw.elapsed (seconds)."""

    def __enter__(self):  # Python calls this at `with`; what it returns goes into `as sw`
        self.start = time.perf_counter()  # (1) remember the time
        return self  # (2) give the caller this very Stopwatch object

    def __exit__(self, exc_type, exc, tb):  # Python calls this when the block ends — even on an error
        self.elapsed = time.perf_counter() - self.start  # (3) seconds since start
        return False  # False = do not swallow an error from the block, let it go on


@contextmanager
def timer(label: str):
    """The same idea written as a generator: code before yield = enter, code after = exit."""
    start = time.perf_counter()
    try:
        yield # (4) hand control to the block inside `with`
    finally:
        print(f"{label}: {time.perf_counter() - start:.2f}s", file=sys.stderr)  # (5) seconds since start

