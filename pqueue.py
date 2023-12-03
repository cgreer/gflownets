from dataclasses import dataclass
from heapq import heapify, heappush, heappop
from itertools import count
from typing import (
    Any,
    Union,
    List,
    Tuple,
)

Number = Union[int, float]
Priority = Number
EntryCount = int # https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes
PriorityItem = Tuple[Priority, Any] # Item added to queue, Any is some task/object you care to track
QueueItem = Tuple[Priority, EntryCount, Any]


@dataclass
class PriorityQueue:
    '''Basic Priority Queue

    Pop will return the item w/ the "highest" priority.

    NOTE! Highest priority means LOWEST int/float priority value. So,
    for e.g., 0 is higher priority than 1.

    If you need to return item with highest "score" instead, then push
    items w/ (-score, item) to make highest scores lowest priority.
    '''
    queue: List[QueueItem]
    entry_counter: Any

    def __len__(self):
        return len(self.queue)

    @classmethod
    def empty(Cls):
        return Cls(queue=[], entry_counter=count())

    @classmethod
    def from_list(Cls, lst: List[PriorityItem]):
        queue = []
        entry_counter = count()
        for priority, thing in lst:
            queue.append((priority, next(entry_counter), thing))
        heapify(queue)
        return Cls(queue=queue, entry_counter=entry_counter)

    def push(self, item: PriorityItem):
        '''
        Add item to queue
        '''
        entry = (item[0], next(self.entry_counter), item[1])
        heappush(self.queue, entry)

    def pop(self):
        '''
        Returns the item w/ highest priority. Popped item is removed
        from queue.

        Items will be returned in order they were added.
        - https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes
        '''
        priority, _, thing = heappop(self.queue)
        return (priority, thing)

    def peek(self):
        '''Return item w/ highest priority, but don't pop it'''
        priority, _, thing = heappop(self.queue)
        return priority, thing


class Tasks:

    def check_pqueue(self):
        from random import randint
        from types import SimpleNamespace as Record
        max_size = 1000

        # Add a bunch of items to pqueue
        pq = PriorityQueue.empty()
        for _ in range(max_size + 1):
            ep = Record()
            ep.reward = randint(0, 100)
            pq.push((ep.reward, ep))

        # Show lowest
        priority, ep = pq.peek()
        print("Peek", priority, ep)

        # Push highest reward on, but keep size
        best_ep = Record()
        best_ep.reward = 1000
        print("Queue Size", len(pq))
        if (len(pq) + 1) > max_size:
            print("queue is full")
            # Higher than lowest?
            if best_ep.reward > priority:
                priority, ep = pq.pop()
                print("Popped", priority, ep)
                pq.push((best_ep.reward, best_ep))
        else:
            # Just add
            pq.push((best_ep.reward, best_ep))

        # Verify highest is in there
        best_reward = max([x[0] for x in pq.queue])
        print("Best", best_reward)
        assert best_reward == 1000


if __name__ == "__main__":
    Tasks().check_pqueue()
