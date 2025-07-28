x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
n = len(x)

mean_x = sum(x) / n
mean_y = sum(y) / n

numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
m = numerator / denominator
b = mean_y - m * mean_x

print(f"Linear Regression Equation: ŷ = {m:.2f}x + {b:.2f}")
predictions = []
for i in range(n):
    y_pred = m * x[i] + b
    predictions.append(y_pred)
    residual = y[i] - y_pred
    print(f"x = {x[i]}, y = {y[i]}, ŷ = {y_pred:.2f}, Residual = {residual:.2f}")

mse = sum((y[i] - predictions[i]) ** 2 for i in range(n)) / n
print(f"\nMean Squared Error (MSE): {mse:.4f}")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = [[1], [2], [3], [4], [5]]  # Note: X must be 2D
y = [2, 4, 5, 4, 5]
model = LinearRegression()
model.fit(X, y)
m = model.coef_[0]  # slope
b = model.intercept_  # intercept
print(f"Linear Regression Equation: ŷ = {m:.2f}x + {b:.2f}")
y_pred = model.predict(X)
for i in range(len(X)):
    residual = y[i] - y_pred[i]
    print(f"x = {X[i][0]}, y = {y[i]}, ŷ = {y_pred[i]:.2f}, Residual = {residual:.2f}")

mse = mean_squared_error(y, y_pred)
print(f"\nMean Squared Error (MSE): {mse:.4f}")
