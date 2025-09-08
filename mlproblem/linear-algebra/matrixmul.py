def matrixmul(a: list[list[int | float]], b: list[list[int | float]]) -> list[list[int | float]]:
    # Check if matrices are compatible
    if len(a[0]) != len(b):
        return -1
    
    # Dimensions
    rows_a = len(a)
    cols_a = len(a[0])
    cols_b = len(b[0])
    
    # Initialize result matrix with zeros
    c = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    # Perform multiplication
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                c[i][j] += a[i][k] * b[k][j]
    
    return c
