def min_max(x: list[int]) -> list[float]:
    min_val = min(x)
    max_val = max(x)
    if min_val == max_val:
        return [0.0 for _ in x]
    normalized = [(val-min_val)/(max_val-min_val) for val in x]
    return normalized
