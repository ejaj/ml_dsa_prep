from typing import List, Dict


def num_to_index(nums: List[int]) -> Dict[int, int]:
    return {num:idx for idx, num in enumerate(nums)}


print(num_to_index([1, 2, 3, 4, 5, 6, 7, 8]))
print(num_to_index([8, 7, 6, 5, 4, 3, 2, 1]))
print(num_to_index([0, 3, 2, 4, 5, 1]))
