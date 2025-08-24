class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        x = init
        for _ in range(iterations):
            grad = 2 * x          # derivative of x^2
            x = x - learning_rate * grad
        return round(x, 5)
