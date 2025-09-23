import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    """
    Calculate the Bhattacharyya distance between two discrete probability distributions.

    Args:
        p (list[float]): First probability distribution.
        q (list[float]): Second probability distribution.

    Returns:
        float: Bhattacharyya distance rounded to 4 decimal places.
    """
    # Convert to numpy arrays for vectorized operations
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    
    # Check for valid input
    if p.size == 0 or q.size == 0 or p.size != q.size:
        return 0.0

    # Calculate Bhattacharyya coefficient (BC)
    bc = np.sum(np.sqrt(p * q))

    # If BC <= 0, distance is infinite (completely different distributions)
    if bc <= 0:
        return float("inf")

    # Calculate Bhattacharyya distance
    distance = -np.log(bc)

    return round(distance, 4)
