from typing import List, Deque
from collections import deque


def rotate_list(arr: List[int], k: int) -> Deque[int]:
    queue = deque(arr)
    for _ in range(k):
        queue.append(queue.popleft())
    return queue



print(rotate_list([1, 2, 3, 4, 5], 0))
print(rotate_list([1, 2, 3, 4, 5], 1))
print(rotate_list([1, 2, 3, 4, 5], 2))
print(rotate_list([1, 2, 3, 4, 5], 3))
print(rotate_list([1, 2, 3, 4, 5], 4))
print(rotate_list([1, 2, 3, 4, 5], 5))
