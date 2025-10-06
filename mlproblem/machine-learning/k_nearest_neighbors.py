from math import sqrt


def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point

    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return

    Returns:
        List of k nearest neighbor points as tuples
    """
    distances = []
    for p in points:
        dist = sqrt(sum((a - b) ** 2 for a, b in zip(p, query_point)))
        distances.append((dist, p))
    distances.sort(key=lambda x: x[0])
    nearest_neighbors = [p for _, p in distances[:k]]
    return nearest_neighbors
