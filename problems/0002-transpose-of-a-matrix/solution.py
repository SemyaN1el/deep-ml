def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
	result = [[0 for _ in range(len(a))] for _ in range(len(a[0]))]
	for i in range(len(a)):
		for j in range(len(a[0])):
			result[j][i] = a[i][j]
	return result