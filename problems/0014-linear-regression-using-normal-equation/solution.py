import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
    X = np.array(X)
    y = np.array(y)
    try:
        theta = np.linalg.inv(X.T @ X) @ X.T @ y
        theta = np.round(theta, 4)
        return theta.tolist() 
    except np.linalg.LinAlgError:
        print("Matrix X^T * X is singular. Cannot compute inverse.")
        return None