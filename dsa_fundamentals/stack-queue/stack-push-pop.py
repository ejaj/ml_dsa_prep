from typing import List


def reverse_list(arr: List[int]) -> List[int]:
    stack = []
    for num in arr:
        stack.append(num)
    reversed_arr = []
    while len(stack) > 0:
        reversed_arr.append(stack.pop())
    return reversed_arr




print(reverse_list([1, 2, 3]))
print(reverse_list([3, 2, 1, 4, 6, 2]))
print(reverse_list([1, 9, 7, 3, 2, 1, 4, 6, 2]))
