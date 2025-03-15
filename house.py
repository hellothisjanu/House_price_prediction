import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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

# Train model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Save model with protocol=4 for compatibility with different Python versions
with open('model.pkl', 'wb') as f:
    pickle.dump(lr, f, protocol=4)

print("Model trained and saved successfully as model.pkl")