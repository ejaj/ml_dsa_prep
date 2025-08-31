from typing import List, Dict
from itertools import zip_longest


def group_names_and_scores(names: List[str], scores: List[int]) -> Dict[str, int]:
    # for name, score in zip_longest(names, scores, fillvalue="Unknown"):
    #     print(f"{name} → {score}")

    return dict(zip(names, scores))


print(group_names_and_scores(["Alice", "Bob", "Charlie"], [90, 80, 70]))
print(group_names_and_scores(["Jane", "Carol", "Charlie"], [25, 100, 60]))
print(group_names_and_scores(["Doug", "Bob", "Tommy"], [80, 90, 100]))
