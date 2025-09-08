import numpy as np

def mae(y_true, y_pred):
	"""
	Calculate Mean Absolute Error between two arrays.

	Parameters:
	y_true (numpy.ndarray): Array of true values
    y_pred (numpy.ndarray): Array of predicted values

	Returns:
	float: Mean Absolute Error rounded to 3 decimal places
	"""
	# Your code here
	y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true-y_pred))
	return round(mae,3)