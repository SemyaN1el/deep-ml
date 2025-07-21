import math
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	
    a,b = matrix[0]
    c,d = matrix[1]
    trace = a + d
    determinant = a*d - b*c
    discriminant = trace ** 2 - 4 * determinant
    sqrt_disc = math.sqrt(discriminant)
    lambda1 = (trace+sqrt_disc)/2
    lambda2 = (trace-sqrt_disc)/2
    return sorted([lambda1,lambda2], reverse = True)
