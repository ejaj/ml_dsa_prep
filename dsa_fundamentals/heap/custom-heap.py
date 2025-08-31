import heapq
from typing import List


def get_reverse_sorted(nums: List[int]) -> List[int]:
    heap = []
    for num in nums:
        heapq.heappush(heap, (-num, num))
    result  = []
    while heap:
        result.append(heapq.heappop(heap)[1])
    return result


print(get_reverse_sorted([1, 2, 3]))
print(get_reverse_sorted([5, 6, 4, 2, 7, 3, 1]))
print(get_reverse_sorted([5, 6, -4, 2, 4, 7, -3, -1]))
