from typing import List

def sort_words(words: List[str]) -> List[str]:
    # Sort words by length in descending order using lambda
    return sorted(words, key=lambda word: len(word), reverse=True)

def sort_numbers(numbers: List[int]) -> List[int]:
    # Sort numbers by absolute value in ascending order using lambda
    return sorted(numbers, key=lambda num: abs(num))

if __name__ == "__main__":
    words = ["apple", "banana", "kiwi", "pear", "watermelon", "blueberry", "cherry"]
    numbers = [5, -3, 8, -1, 0, -7, 2]

    print("Sorted Words:", sort_words(words))
    print("Sorted Numbers:", sort_numbers(numbers))
