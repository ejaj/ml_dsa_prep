import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Data
data = pd.DataFrame({
    "Size": [750, 800, 850, 900, 950, 1000],
    "City": ["New York", "Los Angeles", "Chicago", "New York", "Los Angeles", "Chicago"],
    "Price": [150000, 180000, 200000, 220000, 240000, 260000]
})

X = data[["Size", "City"]]
y = data["Price"]

encoder = OneHotEncoder(drop="first", sparse=False)  # Drop first category to avoid dummy variable trap
X_encoded = encoder.fit_transform(X[["City"]])

X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(["City"]))

X_final = pd.concat([X.drop(columns=["City"]), X_encoded_df], axis=1)
model = LinearRegression()
model.fit(X_final, y)

# Step 3: Predict price for a 875 sqft house in Chicago (Manually encode)
X_test = pd.DataFrame({"Size": [875], "City": ["Chicago"]})
X_test_encoded = encoder.transform(X_test[["City"]])
X_test_final = pd.concat(
    [X_test.drop(columns=["City"]), pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(["City"]))],
    axis=1)

# Predict
predicted_price = model.predict(X_test_final)
print(f"Predicted Price for 875 sqft in Chicago: ${predicted_price[0]:,.2f}")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Data
X = np.array([[750, 2], [800, 3], [850, 3], [900, 4], [950, 4], [1000, 5]])  # [Size, Bedrooms]
y = np.array([150000, 180000, 200000, 220000, 240000, 260000])

# Step 1: Standardize data manually
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Train Linear Regression model
model = LinearRegression()
model.fit(X_scaled, y)

from sklearn.feature_selection import SelectKBest, f_regression

X = np.array(
    [[750, 2, 1], [800, 3, 1], [850, 3, 0], [900, 4, 0], [950, 4, 1], [1000, 5, 1]])  # [Size, Bedrooms, Garage]
y = np.array([150000, 180000, 200000, 220000, 240000, 260000])

selector = SelectKBest(score_func=f_regression, k=2)
X_selected = selector.fit_transform(X, y)
print("Selected Features:\n", X_selected)
print("Feature Scores:\n", selector.scores_)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

X = np.array([[750, 2], [800, 3], [850, 3], [900, 4], [950, 4], [1000, 5]])  # Features: Size, Bedrooms
y = np.array([150000, 180000, 200000, 220000, 240000, 260000])  # House price

rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='r2')
grid_search.fit(X, y)
# Print Best Parameters
print("Best Parameters:", grid_search.best_params_)

best_rf_model = grid_search.best_estimator_
predicted_price = best_rf_model.predict([[875, 3]])
print(f"Predicted Price for 875 sqft with 3 bedrooms: ${predicted_price[0]:,.2f}")

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define parameter distributions (instead of a fixed grid)
param_dist = {
    "n_estimators": randint(50, 200),
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": randint(2, 10)
}

# Randomized Search
random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=10, cv=5, scoring="r2",
                                   random_state=42)
random_search.fit(X, y)

# Print Best Parameters
print("Best Parameters (Random Search):", random_search.best_params_)

# Predict
best_rf_random = random_search.best_estimator_
predicted_price = best_rf_random.predict([[875, 3]])
print(f"Predicted Price for 875 sqft with 3 bedrooms: ${predicted_price[0]:,.2f}")

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample Data
X = np.array([[750, 2], [800, 3], [850, 3], [900, 4], [950, 4], [1000, 5]])  # Features: Size, Bedrooms
y = np.array([150000, 180000, 200000, 220000, 240000, 260000])  # Target: House price

model = LinearRegression()

kf = KFold(n_splits=5, shuffle=True, random_state=5)  # 5 folds

cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

print("Cross-Validation Scores:", cv_scores)
print("Mean R² Score:", np.mean(cv_scores))
print("Standard Deviation of Scores:", np.std(cv_scores))

# Step 3: Predict for 875 sqft, 3 bedrooms (Manually scale input)
X_test = np.array([[875, 3]])
X_test_scaled = scaler.transform(X_test)
predicted_price = model.predict(X_test_scaled)

print(f"Predicted Price for 875 sqft with 3 bedrooms: ${predicted_price[0]:,.2f}")

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Sample Data (Binary Classification: 0 = Cheap, 1 = Expensive)
X = np.array([[750, 2], [800, 3], [850, 3], [900, 4], [950, 4], [1000, 5]])
y = np.array([0, 1, 1, 1, 0, 0])  # Labels: 0 (cheap), 1 (expensive)

# Define Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

# Output
print("Stratified Cross-Validation Accuracy:", cv_scores)
print("Mean Accuracy:", np.mean(cv_scores))
from sklearn.model_selection import LeaveOneOut

# Define LOOCV
loo = LeaveOneOut()
cv_scores = cross_val_score(model, X, y, cv=loo, scoring="accuracy")

# Output
print("LOOCV Scores:", cv_scores)
print("Mean Accuracy:", np.mean(cv_scores))
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
import numpy as np

# Simulated time-series data
X = np.array([[i] for i in range(1, 101)])  # Time feature (1 to 100)
y = np.array([i * 1.5 + np.random.rand() * 5 for i in range(1, 101)])  # Trend-based target

# Time-Series Split (5 folds)
tscv = TimeSeriesSplit(n_splits=5)

# Evaluate Ridge Regression
model = Ridge()
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"R² Score on Fold: {score:.4f}")

import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Sample Data
X = np.random.rand(100, 10)  # 100 samples, 10 features
y = np.random.rand(100) * 1000  # Target variable

# Define objective function for optimization
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 5, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    # 5-fold Cross-validation
    score = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
    return score

# Optimize
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)  # Run 20 trials

# Output Best Hyperparameters
print("Best Hyperparameters:", study.best_params)
