def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	# Your code here
	tp = sum(1 for yt,yp in zip(y_true,y_pred) if yt == yp == 1)
    fp = sum(1 for yt,yp in zip(y_true,y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt,yp in zip(y_true,y_pred) if yt == 1 and yp == 0)
    try:
		precision = tp / (tp + fp)
	except ZeroDivisionError:
		precision = 0.0
    try:
		recall = tp / (tp + fn)
	except ZeroDivisionError:
		recall = 0.0
    try:
		f1 = 2 * (precision * recall) / (precision + recall)
	except ZeroDivisionError:
		f1 = 0.0
	return round(f1,3)