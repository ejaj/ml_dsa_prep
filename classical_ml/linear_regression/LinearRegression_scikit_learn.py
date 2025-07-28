import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Data
X = np.array([750, 800, 850, 900, 950, 1000]).reshape(-1, 1)  # Feature: House size
y = np.array([150000, 180000, 200000, 220000, 240000, 260000])  # Target: House price

# Train Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Predict price for a house with 875 sqft
house_size = np.array([[875]])  # Reshaped to 2D array
predicted_price = model.predict(house_size)

# Output results
print(f"Predicted Price for 875 sqft: ${predicted_price[0]:,.2f}")
print(f"Model Coefficients (slope): {model.coef_[0]}")
print(f"Model Intercept: {model.intercept_}")

# Visualization
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", linestyle="--", label="Regression Line")
plt.scatter(house_size, predicted_price, color="green", marker="o", s=100, label="Prediction (875 sqft)")
plt.xlabel("Size (sqft)")
plt.ylabel("Price ($)")
plt.legend()
plt.title("Linear Regression: House Price Prediction")
plt.show()

print(model.coef_)
print(model.intercept_)

r2_score = model.score(X, y)
print(r2_score)


def mean_squared_error(y, y_pred):
    pass


mse = mean_squared_error(y, predicted_price)
print(f"Mean Squared Error (MSE): {mse:.2f}")
mae = mean_absolute_error(y, predicted_price)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

polynomial_model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
polynomial_model.fit(X, y)
predicted_price = polynomial_model.predict(house_size)
print("Polynomial Regression R²:", polynomial_model.score(X, y))
