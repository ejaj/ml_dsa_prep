class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: float) -> float:
        x = init
        for _ in range(iterations):
            x = x - learning_rate * (2 * x)  # Gradient Descent step
        return round(x, 2)


solution = Solution()
print(solution.get_minimizer(0, 0.01, 5))  # Output: 5
print(solution.get_minimizer(10, 0.01, 5))  # Output: 4.08536
print(solution.get_minimizer(100, 0.1, 10))  # Output: 0.00000
