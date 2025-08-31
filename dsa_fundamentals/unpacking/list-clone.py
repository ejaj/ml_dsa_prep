from typing import List


def remove_element(arr: List[int], element: int) -> List[int]:
    new_list = arr.copy()
    new_list.remove(element)
    return new_list



arr = [1, 3, 5, 7, 9]

print(remove_element(arr, 3))
print(arr)
print(remove_element(arr, 9))
print(arr)
print(remove_element(arr, 1))
print(arr)
