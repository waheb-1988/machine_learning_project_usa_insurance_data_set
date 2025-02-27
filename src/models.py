import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
# Example with DecisionTreeRegressor
decision_tree = DecisionTreeRegressor(random_state=42)

# Example with LinearRegression
linear_regression = LinearRegression()

# Example with Ridge
ridge = Ridge(random_state=42)

# Example with Lasso
lasso = Lasso(random_state=42)

# Example with ElasticNet
elastic_net = ElasticNet(random_state=42)

# Example with SVR
svr = SVR()

# Example with MLPRegressor
mlp = MLPRegressor(random_state=42)

# Example with RandomForestRegressor
random_forest = RandomForestRegressor(random_state=42)

# Example with GradientBoostingRegressor
gradient_boosting = GradientBoostingRegressor(random_state=42)

# Example with AdaBoostRegressor
adaboost = AdaBoostRegressor(random_state=42)

# Example with XGBRegressor
xgb = XGBRegressor(random_state=42)

# Example with LGBMRegressor
lgbm = LGBMRegressor(random_state=42)

# Example with CatBoostRegressor
catboost = CatBoostRegressor(random_state=42)

# Ridge Regressor
ridge_param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}

# Lasso Regressor
lasso_param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100],
    'max_iter': [1000, 5000, 10000]
}

# ElasticNet Regressor
elasticnet_param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
    'max_iter': [1000, 5000, 10000]
}

# Support Vector Regressor (SVR)
svr_param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.2, 0.5],
    'gamma': ['scale', 'auto']
}

# Multi-Layer Perceptron (MLP) Regressor
mlp_param_grid = {
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive']
}

# Decision Tree Regressor
decision_tree_param_grid = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30, 50, 100],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 10],
    'max_features': [None, 'sqrt', 'log2'],
    'max_leaf_nodes': [None, 10, 20, 50, 100]
}

# Random Forest Regressor
random_forest_param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['squared_error', 'absolute_error']
}

# Gradient Boosting Regressor
gradient_boosting_param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 10, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 10],
    'subsample': [0.6, 0.8, 1.0],
    'max_features': ['sqrt', 'log2', None],
    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile']
}

# XGBoost Regressor
xgboost_param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 10, 20],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# LightGBM Regressor
lightgbm_param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [31, 50, 100],
    'max_depth': [-1, 10, 20, 30],
    'min_child_samples': [20, 30, 50],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# CatBoost Regressor
catboost_param_grid = {
    'iterations': [500, 1000, 2000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'depth': [3, 5, 7, 10],
    'l2_leaf_reg': [1, 3, 5, 7],
    'border_count': [32, 64, 128]
}

# AdaBoost Regressor
adaboost_param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 1.0],
    'loss': ['linear', 'square', 'exponential']
}

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

class MLModels:
    # Ridge Regressor
    ridge_param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 100],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }

    # Lasso Regressor
    lasso_param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 100],
        'max_iter': [1000, 5000, 10000]
    }

    # ElasticNet Regressor
    elasticnet_param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 100],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'max_iter': [1000, 5000, 10000]
    }

    # Support Vector Regressor (SVR)
    svr_param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2, 0.5],
        'gamma': ['scale', 'auto']
    }

    # Multi-Layer Perceptron (MLP) Regressor
    mlp_param_grid = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive']
    }

    # Decision Tree Regressor
    decision_tree_param_grid = {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30, 50, 100],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 10],
        'max_features': [None, 'sqrt', 'log2'],
        'max_leaf_nodes': [None, 10, 20, 50, 100]
    }

    # Random Forest Regressor
    random_forest_param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'criterion': ['squared_error', 'absolute_error']
    }

    # Gradient Boosting Regressor
    gradient_boosting_param_grid = {
        'n_estimators': [100, 200, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 10, 20],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 10],
        'subsample': [0.6, 0.8, 1.0],
        'max_features': ['sqrt', 'log2', None],
        'loss': ['squared_error', 'absolute_error', 'huber', 'quantile']
    }

    # XGBoost Regressor
    xgboost_param_grid = {
        'n_estimators': [100, 200, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 10, 20],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }

    # LightGBM Regressor
    lightgbm_param_grid = {
        'n_estimators': [100, 200, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [31, 50, 100],
        'max_depth': [-1, 10, 20, 30],
        'min_child_samples': [20, 30, 50],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # CatBoost Regressor
    catboost_param_grid = {
        'iterations': [500, 1000, 2000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'depth': [3, 5, 7, 10],
        'l2_leaf_reg': [1, 3, 5, 7],
        'border_count': [32, 64, 128]
    }

    # AdaBoost Regressor
    adaboost_param_grid = {
        'n_estimators': [50, 100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 1.0],
        'loss': ['linear', 'square', 'exponential']
    }
    def __init__(self, model_name, params=None):
        self.model_name = model_name
        self.params = params if params else {}
        self.model = self.initialize_model()

    def initialize_model(self):
        models = {
            'decision_tree': DecisionTreeRegressor,
            'linear_regression': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'elastic_net': ElasticNet,
            'svr': SVR,
            'mlp': MLPRegressor,
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'adaboost': AdaBoostRegressor,
            'xgb': XGBRegressor,
            'lgbm': LGBMRegressor,
            'catboost': CatBoostRegressor
        }
        if self.model_name in models:
            return models[self.model_name](**self.params)
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")

    def get_model(self):
        return self.model

# Usage Example:
# model = MLModels('xgb', {'random_state': 42}).get_model()
# print(model)
