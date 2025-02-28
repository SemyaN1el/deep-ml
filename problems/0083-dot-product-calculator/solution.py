import numpy as np

def calculate_dot_product(vec1, vec2) -> float:
	ans = np.sum(vec1 * vec2)
	return ans