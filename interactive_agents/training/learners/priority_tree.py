"""
Unusual data structure used for prioritized replay.  Allows us to quickly
sample from the priority-weighted distribution over instances.
"""
import numpy as np

class PriorityTree:
    """
    Data structure used to efficiently sample from a priority weighted
    distribution, and to compute importance weights.
    """

    def __init__(self, capacity):
        self._capacity = 1
        self._depth = 0

        while self._capacity < capacity:
            self._capacity *= 2
            self._depth += 1

        size = self._capacity * 2
        self._sums = np.full(size, 0.0)
        self._mins = np.full(size, np.inf)

        self._next_index = 0

    def set(self, indices, priorities):
        priorities = np.asarray(priorities)
        indices = np.asarray(indices, dtype=np.int64)
        indices += self._capacity

        self._sums[indices] = priorities
        self._mins[indices] = priorities

        for _ in range(self._depth):
            indices //= 2
            left = indices * 2
            right = left + 1
            self._sums[indices] = self._sums[left] + self._sums[right]
            self._mins[indices] = np.minimum(self._mins[left], self._mins[right])

    def get(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return self._sums[indices + self._capacity]

    def min(self):
        """Returns the min over all values"""
        return self._mins[1]

    def sum(self):
        """Returns the sum of all values"""
        return self._sums[1]

    def prefix_index(self, prefix):
        """
        Returns the largest index i such that the sum a[0] + ... + a[i-1] <= prefix
        """

        idx = 1
        for _ in range(self._depth):
            next_idx = idx * 2
            if prefix < self._sums[next_idx]:
                idx = next_idx
            else:
                prefix -= self._sums[next_idx]
                idx = next_idx + 1
        
        return idx - self._capacity


# UNIT TESTS
def test_priority_tree():
    tree = PriorityTree(7)
    assert tree._capacity == 8

    tree.set([0], [1])
    assert tree.min() == 1
    assert tree.sum() == 1
    assert tree.prefix_index(0.5) == 0

    tree.set([0, 2, 3, 4], [2, 1, 4, 3])
    assert tree.min() == 1
    assert tree.sum() == 10
    assert tree.prefix_index(1) == 0
    assert tree.prefix_index(2) == 2
    assert tree.prefix_index(6) == 3
    assert tree.prefix_index(11) == 7

    tree.set([1, 5, 6, 7], [.5, 2, 1, 3])
    assert tree.min() == .5
    assert tree.sum() == 16.5
    assert tree.prefix_index(1) == 0
    assert tree.prefix_index(7) == 3
    assert tree.prefix_index(13) == 6
    assert tree.prefix_index(20) == 7
