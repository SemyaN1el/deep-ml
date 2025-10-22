import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
    data = np.array(vectors)
    cov_matrix = np.cov(data)
	return cov_matrix.tolist()