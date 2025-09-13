from collections import Counter
def disorder(apples: list) -> float:
	"""
	Compute the disorder in a basket of apples.
	"""
	# Your code here
	counts = Counter(apples)
    n = len(apples)
    probs = sum((count/n)**2 for count in counts.values())
    gini = 1 - probs
    return gini
    