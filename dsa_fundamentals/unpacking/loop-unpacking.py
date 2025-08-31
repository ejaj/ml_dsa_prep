from typing import List, Tuple


def best_student(scores: List[Tuple[str, int]]) -> str:
    best_name = ""
    best_score = -1
    for name, score in scores:
        if scores > best_score:
            best_score = score
            best_name = name
    return best_name


print(best_student([("Alice", 90), ("Bob", 80), ("Charlie", 70)]))
print(best_student([("Alice", 90), ("Bob", 80), ("Charlie", 100)]))
print(best_student([("Alice", 90), ("Bob", 100), ("Charlie", 70)]))
print(best_student([("Alice", 90), ("Bob", 90), ("Charlie", 80), ("David", 100)]))
