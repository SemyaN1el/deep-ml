def hard_sigmoid(x: float) -> float:
	"""
	Implements the Hard Sigmoid activation function.

	Args:
		x (float): Input value

	Returns:
		float: The Hard Sigmoid of the input
	"""
	# Your code here
	return 0.0 if x <= -2.5 else 0.2*x + 0.5 if -2.5 < x < 2.5 else 1.0