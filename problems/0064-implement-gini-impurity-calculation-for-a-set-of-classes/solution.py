
import numpy as np
from collections import Counter
def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
    counts = Counter(y)
    total = len(y)
    val = 1 - sum((count/total)**2 for count in counts.values())
	return round(val,3)