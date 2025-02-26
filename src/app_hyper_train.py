
from data_import import read_file
from hyper_model import hyperparameter_tuning

# Read Data
df = read_file('insurance.csv')


# Linear Regression
linear_param_grid = {
    'fit_intercept': [True, False],
    'normalize': [True, False]
}

random_forest_param_grid = {
    'model__n_estimators': [100, 200, 500],
    'model__max_depth': [None, 10, 20, 30, 50],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['auto', 'sqrt', 'log2'],
    'model__criterion': ['squared_error', 'absolute_error']
}



# XGBoost
xgboost_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [1.0, 0.1, 0.01]
}

# Read Data
df = read_file('insurance.csv')
# Example Usage for Linear Regression
#hyperparameter_tuning('linear', linear_param_grid,df)

# Example Usage for Random Forest
hyperparameter_tuning('random_forest', random_forest_param_grid,df)

# Example Usage for XGBoost
# hyperparameter_tuning('xgboost', xgboost_param_grid,df)
