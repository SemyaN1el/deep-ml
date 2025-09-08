import numpy as np

def accuracy_score(y_true, y_pred):
	# Your code here
	correct = sum(1 for yt,yp in zip(y_true,y_pred) if yt == yp)
    return correct / len(y_true)