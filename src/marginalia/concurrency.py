"""Run a slow function over many inputs at once — for the 40 golden-set questions in week 3.

Exercise: fill in the three `...` below, then run
    uv run pytest tests/test_concurrency.py -v
"""

import asyncio
from collections.abc import Callable


async def run_all(func: Callable, inputs: list, max_concurrent: int = 4) -> list:
    """Call func(x) for every x in inputs, at most max_concurrent at a time. Results keep the input order."""
    gate = asyncio.Semaphore(max_concurrent)  # a gate with N slots: at most N tasks inside at once

    async def run_one(x):
        async with gate:  # wait for a free slot (an async context manager)
            # func is ordinary blocking code (our agent is), so run it in a thread and await the result:
            return await asyncio.to_thread(func, x)

    # (2) start run_one(x) for EVERY input at once and wait for all of them — like Promise.all:
    return await asyncio.gather(*(run_one(x) for x in inputs))


def run_all_sync(func: Callable, inputs: list, max_concurrent: int = 4) -> list:
    """The same for ordinary (non-async) code: start an event loop, run run_all in it, return the results."""
    return asyncio.run(run_all(func, inputs, max_concurrent))
