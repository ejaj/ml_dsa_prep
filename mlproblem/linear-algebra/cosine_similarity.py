import numpy as np

def cosine_similarity(v1, v2):
    if v1.shape != v2.shape:
        raise ValueError("Vectors must have the same shape")
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("Vectors must not be zero vectors")
    
    # Calculate dot product
    dot_product = np.dot(v1, v2)
    # Apply formula
    similarity = dot_product / (norm_v1 * norm_v2)
    return round(similarity, 3)