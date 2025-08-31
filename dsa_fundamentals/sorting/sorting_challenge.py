from typing import List
def sort_words(words:List[str]) -> List[str]:
    return sorted(words, key=len, reverse=True)
def sort_numbers(numers:List[int]) -> List[int]:
    return sorted(numers, key=abs)
if __name__ == "__main__":
    words = ["apple", "banana", "kiwi", "pear", "watermelon", "blueberry", "cherry"]
    numbers = [5, -3, 8, -1, 0, -7, 2]

    print("Sorted Words:", sort_words(words))
    print("Sorted Numbers:", sort_numbers(numbers))