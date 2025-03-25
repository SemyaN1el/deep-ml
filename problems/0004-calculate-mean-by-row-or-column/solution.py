def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if not matrix or mode not in ['column','row']:
        return []
    if mode == 'row':
        return [sum(row) / len(row) for row in matrix ]
    if mode == 'column':
        transposed = list(zip(*matrix))
        return [sum(col)/len(col) for col in transposed]