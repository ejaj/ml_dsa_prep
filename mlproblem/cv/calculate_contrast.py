import numpy as np

def calculate_contrast(img: np.ndarray) -> int:
    if img.size == 0:
        return -1
    if np.any(img<0) or np.any(img > 255):
        return -1
    max_val = np.max(img)
    min_val = np.min(img)

    contrast = max_val - min_val
    return int(contrast)