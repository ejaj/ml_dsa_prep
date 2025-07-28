def function_a(n):
    if n == 0:
        return 1
    else:
        return n * funcion_b(n-1)
def funcion_b(n):
    if n == 0:
        return 1
    else:
        return n * function_a(n-1)

function_a(4)