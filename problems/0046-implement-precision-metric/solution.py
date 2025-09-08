import numpy as np
def precision(y_true, y_pred):
	# Your code here
    tp = sum(1 for yt,yp in zip(y_true,y_pred) if yt == yp == 1)
    fp = sum(1 for yt,yp in zip(y_true,y_pred) if yt == 0 and yp == 1)
    return tp / (tp + fp)
