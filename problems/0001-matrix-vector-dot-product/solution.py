def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float]:
    if len(a[0]) != len(b):
        return -1
    vals = []
    for i in a:
        summa = 0
        for j in range(len(i)):
            summa+=(i[j] * b[j])
        vals.append(summa)

    return vals