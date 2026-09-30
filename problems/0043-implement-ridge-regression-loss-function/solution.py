import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	# Your code here
	predictions = X @ w 
	mse = np.mean((predictions - y_true) ** 2)
	regularization = alpha * np.sum(w ** 2)
	return float(mse + regularization)