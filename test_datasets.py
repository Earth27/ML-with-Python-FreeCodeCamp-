import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras import models, layers

# Assuming df is already loaded
# Separate target
X = df.drop('expenses', axis=1)
y = df['expenses']

# Identify categorical columns
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64','float64']).columns.tolist()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Apply preprocessing to train and test datasets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Create a simple linear regression model
model = models.Sequential([
    layers.Dense(1, input_dim=X_train_processed.shape[1])
])

model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])

# Train the model
model.fit(X_train_processed, y_train, epochs=100, batch_size=32, verbose=0)

# Evaluate the model
loss, mae = model.evaluate(X_test_processed, y_test)
print("Mean Absolute Error on test data:", mae)

# Predict and plot
import matplotlib.pyplot as plt

y_pred = model.predict(X_test_processed).flatten()

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Expenses")
plt.ylabel("Predicted Expenses")
plt.title("Actual vs Predicted Healthcare Expenses")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
plt.show()
