from sklearn.ensemble import RandomForestRegressor
from data_import import read_file
from hyper_model import hyperparameter_tuning

# Read Data
df = read_file('insurance.csv')

# Define parameter grid for Random Forest
random_forest_param_grid = {
    'model__n_estimators': [100, 200, 500],
    'model__max_depth': [None, 10, 20, 30, 50],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2', None],  # Fixed here
    'model__criterion': ['squared_error', 'absolute_error']
}

# Call the hyperparameter tuning function
hyperparameter_tuning(RandomForestRegressor(random_state=42), random_forest_param_grid, df)
