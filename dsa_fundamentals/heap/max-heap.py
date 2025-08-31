import heapq
from typing import List


def get_reverse_sorted(nums: List[int]) -> List[int]:
    max_heap = [-x for x in nums]
    heapq.heapify(max_heap)
    result = []
    while max_heap:
        result.append(-heapq.heappop(max_heap))
    return result





print(get_reverse_sorted([1, 2, 3]))
print(get_reverse_sorted([5, 6, 4, 2, 7, 3, 1]))
print(get_reverse_sorted([5, 6, -4, 2, 4, 7, -3, -1]))
