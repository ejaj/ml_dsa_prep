from typing import List


def sort_words(words: List[str]) -> List[str]:
    words.sort()
    return words

def sort_numbers(numbers: List[int]) -> List[int]:
    numbers.sort()
    return numbers

def sort_decimals(numbers: List[float]) -> List[float]:
    numbers.sort()
    return numbers
def sort_decimals_reverse(numbers: List[float]) -> List[float]:
    # Sort the list of floating-point numbers in descending order
    numbers.sort(reverse=True)
    return numbers

words = ["grape", "apple", "banana", "orange"]
print(sort_words(words))
# ['apple', 'banana', 'grape', 'orange']

numbers = [5, 3, 6, 2, 1]
print(sort_numbers(numbers))
# [1, 2, 3, 5, 6]

decimals = [3.14, 2.71, 1.41, 4.67]
print(sort_decimals(decimals))
# [1.41, 2.71, 3.14, 4.67]