def determinant_4x4(matrix: list[list[int|float]]) -> float:
	# Your recursive implementation here
	if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    n = len(matrix)
    for j in range(n):
        sign = (-1)**j
        minor = []
        for i in range(1,n):
            row = []
            for k in range(n):
                if k != j:
                    row.append(matrix[i][k])
            minor.append(row)
        det += sign * matrix[0][j] * determinant_4x4(minor)
    return det


    