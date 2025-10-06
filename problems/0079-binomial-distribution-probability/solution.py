import math

def binomial_probability(n, k, p):
	"""
    Calculate the probability of achieving exactly k successes in n independent Bernoulli trials,
    each with probability p of success, using the Binomial distribution formula.
    """
	# Your code here
    if p == 0:
        return 1.0 if k == 0 else 0.0
    if p == 1:
        return 1.0 if k == n else 0.0
    
    # Calculate binomial coefficient C(n, k) = n! / (k! * (n-k)!)
    binomial_coeff = math.comb(n, k)
    
    # Calculate probability
    probability = binomial_coeff * (p ** k) * ((1 - p) ** (n - k))
	return round(probability, 5)