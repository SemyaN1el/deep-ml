import numpy as np

def to_categorical(x, n_col=None):
	if n_col == None:
        n_col = np.max(x) + 1
    one_hot = np.zeros((len(x),n_col))
    for i in range(len(x)):
        one_hot[i, x[i]] = 1
    return one_hot
    

	