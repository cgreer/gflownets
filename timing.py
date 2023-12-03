from collections import defaultdict
from dataclasses import dataclass
import time
from rich import print as rprint


@dataclass
class LoopInfo:
    start_time: float = None
    last_split_time: float = None
    counter: int = 0

    def __post_init__(self):
        t = time.time()
        self.start_time = t
        self.last_split_time = t


TIMING_LOOP_DATA = defaultdict(LoopInfo)


def format_rate(rate):
    if rate < 1.0:
        rate = round(rate, 4)
    elif rate < 100.0:
        rate = round(rate, 2)
    else:
        rate = round(rate)
    rate = "{:,}".format(rate)
    return rate


def format_count(count):
    return "{:,}".format(count)


def report_every(
    loop_key,
    every_n=1000,
    elements_per_call=1,
    extra=None,
):
    # XXX Fix split
    loop_info = TIMING_LOOP_DATA[loop_key]
    loop_info.counter += 1
    num_completed = (loop_info.counter + 1) * elements_per_call
    if num_completed % every_n == 0:
        now = time.time()
        elapsed = now - loop_info.start_time
        rate = num_completed / elapsed

        split_elapsed = now - loop_info.last_split_time
        split_rate = (every_n) / split_elapsed
        loop_info.last_split_time = now

        rate = format_rate(rate)
        split_rate = format_rate(split_rate)
        if extra:
            extra = [f"{key}:{val}" for key, val in extra.items()]
        else:
            extra = []
        rprint(
            loop_key,
            format_count(num_completed),
            f"split:{split_rate}",
            f"total:{rate}",
            *extra
        )

    # useful to return to know when fired
    return (num_completed % every_n) == 0


class Timer:

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

    def report(self):
        print("elapsed:", f"{round(self.interval, 6)}s")

    def rate(self, num_items, rounding=4):
        rate = num_items / self.interval
        return round(rate, rounding)


class report_elapsed:

    def __init__(self, message=None):
        self.message = message

    def __enter__(self):
        if self.message:
            print(self.message)
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print("elapsed:", f"{round(self.interval, 6)}s")
