import numpy as np
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
	matrix = np.array(matrix, dtype = np.float64)
    determinant = np.linalg.det(matrix)

    return np.linalg.inv(matrix).tolist()