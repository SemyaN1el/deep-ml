import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here
	mean = np.mean(data,axis = 0)
	std_dev = np.std(data,axis = 0)
	stand_data = (data-mean)/std_dev
	stand_data = np.round(stand_data,4)

	min_vals = np.min(data,axis=0)
	max_vals = np.max(data,axis=0)
	norm_data = (data-min_vals)/(max_vals-min_vals)
	norm_data = np.round(norm_data,4)

	return stand_data, norm_data