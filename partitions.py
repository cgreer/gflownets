from typing import (
    Dict,
)
from dataclasses import dataclass

enu = enumerate

NEG_INF = float("-inf")
POS_INF = float("inf")


@dataclass
class Endpoint:
    value: float
    closed: bool

    @property
    def open(self):
        return not self.closed

    def copy(self):
        return Endpoint(
            value=self.value,
            closed=self.closed,
        )


@dataclass
class Range:
    left: Endpoint = None
    right: Endpoint = None

    def __post_init__(self):
        if self.left is None:
            self.left = Endpoint(value=NEG_INF, closed=True)
        if self.right is None:
            self.right = Endpoint(value=POS_INF, closed=True)

    def copy(self):
        return Range(
            left=self.left.copy(),
            right=self.right.copy(),
        )

    def __str__(self):
        s = "[" if self.left.closed else "("
        s += str(self.left.value) + ", " + str(self.right.value)
        s += "]" if self.right.closed else "]"
        return s


@dataclass
class Partition:
    ranges: Dict[int, Range] = None

    def __post_init__(self):
        if self.ranges is None:
            self.ranges = {}

    def copy(self):
        ranges = {}
        for k, v in self.ranges.items():
            ranges[k] = v.copy()
        return Partition(ranges=ranges)

    def __getitem__(self, key):
        return self.ranges[key]

    def __setitem__(self, key, value):
        self.ranges[key] = value

    def __contains__(self, key):
        return key in self.ranges

    def __str__(self):
        s = ""
        for i, key in enu(sorted(self.ranges.keys())):
            spacer = " " if i > 0 else ""
            s += spacer + str(key) + ":" + str(self.ranges[key])
        return s


class Tasks:

    def check_partitions(self):
        # root partition
        part = Partition()
        feature = 5
        thresh = 0.70

        # Pass split child
        pass_part = part.copy()
        if feature not in part:
            pass_part[feature] = Range()
        pass_part[feature].right.value = thresh
        pass_part[feature].right.closed = True

        # Fail child split
        fail_part = part.copy()
        if feature not in part:
            fail_part[feature] = Range()
        fail_part[feature].left.value = thresh
        fail_part[feature].left.closed = False

        print(part)
        print(pass_part)
        print(fail_part)


if __name__ == "__main__":
    Tasks().check_partitions()
