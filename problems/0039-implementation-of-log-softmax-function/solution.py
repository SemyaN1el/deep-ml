import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
    scores = np.array(scores)
	return scores - np.max(scores) - np.log(np.sum(np.exp(scores-np.max(scores))))