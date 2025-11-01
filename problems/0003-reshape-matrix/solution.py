import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
    a = np.array(a)
    if a.shape[0] * a.shape[1] != new_shape[0] * new_shape[1]:
        return []
	reshaped_matrix = np.reshape(a,new_shape)
    return [reshaped_matrix]