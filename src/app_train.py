from data_import import read_file
from train_model import train_and_evaluate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pipeline import build_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import time     
from sklearn.model_selection import train_test_split
import joblib
import os
# Read Data
df = read_file('insurance.csv')

# Use with Linear Regression
# lr_pipeline = build_pipeline(LinearRegression())
# train_and_evaluate(lr_pipeline, df)
from sklearn.pipeline import Pipeline
# Use with Random Forest
# Build preprocessing and model pipelines separately
preprocessor = build_pipeline(None)  # Pass None to get only the preprocessing part
model_pipeline = Pipeline(steps=[
    ('model', RandomForestRegressor(random_state=20))
])

# Train and evaluate
train_and_evaluate(preprocessor, model_pipeline, df)

# Use with Random Forest
# xg_pipeline = build_pipeline(XGBRegressor(objective='reg:squarederror', random_state=42))
# train_and_evaluate(xg_pipeline, df)

# Function to Save Model
# def save_model(model, model_name):
#     """
#     Saves the trained model as a .pkl file in the 'models' folder.
    
#     Parameters:
#         model (sklearn estimator): The trained model to save.
#         model_name (str): The name of the model for the file.
        
#     Returns:
#         None: Prints confirmation message upon successful saving.
#     """
    
#     # Create the 'models' folder if it doesn't exist
#     model_dir = './models'
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)
    
#     # Construct the file path
#     file_path = os.path.join(model_dir, f"{model_name}.pkl")
    
#     # Save the model as a .pkl file
#     joblib.dump(model, file_path)
    
#     # Confirmation message
#     print(f"Model saved successfully as '{file_path}'")
    

# save_model(rf_pipeline, "rf_pipeline")