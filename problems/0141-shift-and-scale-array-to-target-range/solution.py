import numpy as np

def convert_range(x: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.
    """
    # Your code here
    a = np.min(x)
    b = np.max(x)
    if a == b:
        return np.full_like(x, c, dype=float)
    result = c + (d-c)*(x-a)/(b-a)
    return result.astype(float)

    
    