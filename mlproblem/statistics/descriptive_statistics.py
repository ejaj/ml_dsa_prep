import numpy as np

def descriptive_statistics(data):
    if data is None:
        raise ValueError("data cannot be None")
    arr = np.asarray(list(data), dtype=float)
    if arr.size == 0:
        raise ValueError("data must contain at least one value")
    
    # Mean & median
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    
    # Mode (smallest value among those with max frequency)
    vals, counts = np.unique(arr, return_counts=True)
    max_count = counts.max()
    mode_val = vals[counts == max_count].min()
    # If the original data were all integers, return int for mode; else float
    mode = int(mode_val) if np.all(np.mod(arr, 1) == 0) else float(mode_val)
    
    # Variance & standard deviation (population, ddof=0) rounded to 4 decimals
    variance = round(float(np.var(arr, ddof=0)), 4)
    std_dev = round(float(np.std(arr, ddof=0)), 4)
    
    # Percentiles & IQR
    p25, p50, p75 = np.percentile(arr, [25, 50, 75])
    iqr = float(p75 - p25)
    
    return {
        "mean": mean,
        "median": median,
        "mode": mode,
        "variance": variance,
        "standard_deviation": std_dev,
        "25th_percentile": float(p25),
        "50th_percentile": float(p50),
        "75th_percentile": float(p75),
        "interquartile_range": iqr
    }

# Example
# data = [10, 20, 30, 40, 50]
# print(descriptive_statistics(data))
