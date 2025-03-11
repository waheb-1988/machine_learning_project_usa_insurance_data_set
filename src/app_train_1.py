import os
import joblib
from data_import import read_file
from train_model import train_and_evaluate
from sklearn.ensemble import RandomForestRegressor
from pipeline import build_pipeline
from sklearn.pipeline import Pipeline
from models import MLModels
# Read Data
df = read_file('insurance.csv')

# Build pipeline with preprocessing and model
preprocessor = build_pipeline(None)
# Initialize the ML model with desired parameters
model_name = 'gradient_boosting'  # Using Gradient Boosting Regressor
model_params = {
    'random_state': 20,
   }  # Example params
ml_model = MLModels(model_name, model_params).get_model()

# Create pipeline with the selected model
model_pipeline = Pipeline(steps=[
    ('model', ml_model)
])

# Train and evaluate
train_and_evaluate(preprocessor, model_pipeline, df)
# Train and evaluate



