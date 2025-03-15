import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split

# Load dataset
df = pd.read_csv("house_data.csv")

# Define relevant columns
columns = ['bedrooms', 'bathrooms', 'floors', 'yr_built', 'price']
df = df[columns]

# Define features and target
X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

# Convert target variable to 1D array for compatibility
y = np.ravel(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Hyperparameter tuning using GridSearchCV for Ridge Regression
ridge_param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge = Ridge()
ridge_grid_search = GridSearchCV(ridge, ridge_param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_grid_search.fit(X_train, y_train)

# Hyperparameter tuning using RandomizedSearchCV for Random Forest Regression
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestRegressor()
rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

# Select best model based on GridSearchCV
best_model = ridge_grid_search.best_estimator_ if ridge_grid_search.best_score_ > rf_grid_search.best_score_ else rf_grid_search.best_estimator_

# Save best model with protocol=4 for compatibility with different Python versions
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f, protocol=4)

print("Best model trained and saved successfully as model.pkl")
