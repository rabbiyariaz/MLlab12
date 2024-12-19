import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Generate random data
X = np.random.rand(100, 1) * 10  # Random data
y = 2.5 * X.flatten() + np.random.randn(100) * 2  # Linear relationship with noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    # Define and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Log parameters
    mlflow.log_param("fit_intercept", model.fit_intercept)

    # Predict and log metrics
    y_pred = model.predict(X_test)
    
    # Manually compute RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mlflow.log_metric("rmse", rmse)

    # Log the model
    mlflow.sklearn.log_model(model, "linear_model")
