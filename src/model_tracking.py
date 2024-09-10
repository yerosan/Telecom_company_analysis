import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load your dataset
class PredictionModel():
    def __init__(self, data):
        self.data=data

    def prediction(self):

        data = pd.DataFrame(self.data)

        X = data[['engagement_score', 'experience_score']]
        y = data['satisfaction_score']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define model
        model = RandomForestRegressor()

        # Start tracking with MLflow
        mlflow.start_run()

        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)

        # Train model
        model.fit(X_train, y_train)

        # Log metrics
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        
        mlflow.log_metric("mse", mse)

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        # End the run
        mlflow.end_run()
        return predictions,mse
