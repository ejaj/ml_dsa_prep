import numpy as np

def newtons_method(gradient_fn, hessian_fn, x_init, iterations=10):
    """
    Implements Newton's Method for optimization.

    **Mathematical Formula:**
    -------------------------
    Update rule:
        x_new = x_old - H^{-1} * ∇f(x)

    Where:
    - ∇f(x) is the first derivative (gradient).
    - H is the Hessian matrix (second derivative).
    - H^{-1} is the inverse of the Hessian.

    **Parameters:**
    - gradient_fn (function): Function to compute gradient.
    - hessian_fn (function): Function to compute Hessian matrix.
    - x_init (float): Initial guess for x.
    - iterations (int): Number of iterations.

    **Returns:**
    - float: Optimized value of x.

    **Example Usage:**
    #gradient_fn = lambda x: 2 * x  # f(x) = x^2 → f'(x) = 2x
    #hessian_fn = lambda x: 2       # f''(x) = 2 (constant for quadratic functions)
    #newtons_method(gradient_fn, hessian_fn, x_init=5, iterations=5)
    0.0
    """
    x = x_init  # Initial guess

    print(f"Initial x: {x}")

    for i in range(iterations):
        grad = gradient_fn(x)  # Compute gradient
        hess = hessian_fn(x)   # Compute Hessian (second derivative)
        x -= grad / hess  # Newton's update

        # Print x at each iteration for tracking
        print(f"Iteration {i+1}: x = {x}, Gradient = {grad}, Hessian = {hess}")

    return round(x, 5)

gradient_fn = lambda x: 2 * x  # f(x) = x^2 → f'(x) = 2x
hessian_fn = lambda x: 2       # f''(x) = 2 (constant)

print(newtons_method(gradient_fn, hessian_fn, x_init=5, iterations=5))  # Output: 0.0
