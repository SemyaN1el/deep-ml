import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	# Your code here
	entropy = -sum(pi*np.log(pi) for pi in P if pi != 0)
    cross_entropy = -sum(pi*np.log(qi) for pi,qi in zip(P,Q))
    return (abs(entropy), abs(cross_entropy))