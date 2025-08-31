from typing import List

def sort_words(words: List[str]) -> List[str]:
    # Sort words in descending lexicographical order
    words.sort(reverse=True)
    return words

def sort_numbers(numbers: List[int]) -> List[int]:
    # Sort numbers in descending order
    numbers.sort(reverse=True)
    return numbers

def sort_decimals(numbers: List[float]) -> List[float]:
    # Sort decimal numbers in descending order
    numbers.sort(reverse=True)
    return numbers

print(sort_words(["cherry", "apple", "blueberry", "banana", "watermelon", "zucchini", "kiwi", "pear"]))
# ['zucchini', 'watermelon', 'pear', 'kiwi', 'cherry', 'blueberry', 'banana', 'apple']

print(sort_numbers([1, 5, 3, 2, 4, 11, 19, 9, 2, 5, 6, 7, 4, 2, 6]))
# [19, 11, 9, 7, 6, 6, 5, 5, 4, 4, 3, 2, 2, 2, 1]

print(sort_decimals([3.14, 2.82, 6.433, 7.9, 21.555, 21.554]))
# [21.555, 21.554, 7.9, 6.433, 3.14, 2.82]
