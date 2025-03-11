from sklearn.ensemble import RandomForestRegressor
from data_import import read_file
from hyper_model import hyperparameter_tuning
from models import MLModels
# Read Data
df = read_file('insurance.csv')

# Create an instance of the MLModels class for Ridge Regressor
ml_model = MLModels('gradient_boosting')

# Get the Ridge model
model = ml_model.get_model()

# Access the parameter grid for Ridge Regressor from the MLModels class
param_grid = ml_model.gradient_boosting_param_grid

# Call the hyperparameter tuning function
hyperparameter_tuning(model, param_grid, df)
