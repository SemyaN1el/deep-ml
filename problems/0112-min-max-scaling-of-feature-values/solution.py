import numpy as np

def min_max(x: list[int]) -> list[float]:
    xmin = np.min(x)
    xmax = np.max(x)
    if xmax-xmin == 0:
        return [0.0 for _ in x]
    return [(val - xmin) / (xmax-xmin) for val in x]