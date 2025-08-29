import math

def softmax(scores: list[float]) -> list[float]:
	# Your code here
    exp_scores = [math.exp(score) for score in scores]
    sum_exp_scores = sum(exp_scores)
    probabilities = [round(exp_score / sum_exp_scores,4) for exp_score in exp_scores]
	return probabilities