import numpy as np
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pipeline import build_pipeline

def hyperparameter_tuning(model_name, param_grid, df, target_column="charges", test_size=0.25, random_state=20):
    """
    Hyperparameter tuning and model evaluation function.
    
    Parameters:
        model_name (str): The name of the model ('linear', 'random_forest', 'xgboost')
        param_grid (dict): The hyperparameter grid for GridSearchCV
        df (pd.DataFrame): The dataset including features and target column
        target_column (str): The target variable for prediction
        test_size (float): Test set size as a fraction
        random_state (int): Seed for reproducibility
        
    Returns:
        None: Prints model performance and best parameters
    """
    
    # Model Selection
    if model_name == 'linear':
        model = LinearRegression()
    elif model_name == 'random_forest':
        model = RandomForestRegressor(random_state=42)
    elif model_name == 'xgboost':
        model = XGBRegressor(objective='reg:squarederror', random_state=42)
    else:
        raise ValueError("Invalid model_name. Choose from 'linear', 'random_forest', or 'xgboost'.")
    
    
    # Data Preparation
    X = df.drop(columns=target_column)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Build the complete pipeline
    pipeline = build_pipeline(model)
      
    # Grid Search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    

    # Timing training
    training_start = time.perf_counter()
    grid_search.fit(X_train, y_train)
    training_end = time.perf_counter()
    
    best_model = grid_search.best_estimator_
    
    # Timing prediction
    prediction_start = time.perf_counter()
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    prediction_end = time.perf_counter()
    
    # Metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    train_time = training_end - training_start
    prediction_time = prediction_end - prediction_start
    
    # Output results
    print(f"\n{model_name.upper()} Model Performance:")
    print("\nTraining Set:")
    print(f"  - RMSE: {rmse_train:.4f}")
    print(f"  - MAE: {mae_train:.4f}")
    print(f"  - R^2 Score: {r2_train:.4f}")
    print(f"  - Training Time: {train_time:.4f} seconds")

    print("\nTesting Set:")
    print(f"  - RMSE: {rmse_test:.4f}")
    print(f"  - MAE: {mae_test:.4f}")
    print(f"  - R^2 Score: {r2_test:.4f}")
    print(f"  - Prediction Time: {prediction_time:.5f} seconds")
    print(f"  - Best Parameters: {grid_search.best_params_}")






# # Define the parameter grid for hyperparameter tuning
# random_forest_param_grid = {
#     'model__n_estimators': [100, 200, 500],
#     'model__max_depth': [None, 10, 20, 30, 50],
#     'model__min_samples_split': [2, 5, 10],
#     'model__min_samples_leaf': [1, 2, 4],
#     'model__max_features': ['auto', 'sqrt', 'log2'],
#     'model__criterion': ['squared_error', 'absolute_error']
# }

# # Initialize Random Forest Regressor
# rf = RandomForestRegressor(random_state=42)

# # Initialize GridSearchCV
# grid_search = GridSearchCV(
#     estimator=RandomForestRegressor(random_state=20), 
#     param_grid=random_forest_param_grid, 
#     scoring='neg_mean_squared_error',
#     cv=5,  # 5-fold cross-validation
#     verbose=2,
#     n_jobs=-1
# )

# # Timing the hyperparameter tuning process
# tuning_start = time.perf_counter()
# grid_search.fit(X_train, y_train)
# tuning_end = time.perf_counter()

# # Get the best hyperparameters
# best_params = grid_search.best_params_
# print("Best Hyperparameters:", best_params)

# # Train the model using the best hyperparameters
# best_model = grid_search.best_estimator_

# # Timing the training process
# training_start = time.perf_counter()
# best_model.fit(X_train, y_train)
# training_end = time.perf_counter()

# # Timing the prediction process
# prediction_start = time.perf_counter()
# y_pred_train = best_model.predict(X_train)
# y_pred_test = best_model.predict(X_test)
# prediction_end = time.perf_counter()

# # Calculate KPIs
# # Root Mean Squared Error (RMSE)
# rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
# rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# # Mean Absolute Error (MAE)
# mae_train = mean_absolute_error(y_train, y_pred_train)
# mae_test = mean_absolute_error(y_test, y_pred_test)

# # R-squared score
# r2_train = r2_score(X_train, y_pred_train)
# r2_test = r2_score(y_test, y_pred_test)

# # Time metrics
# train_time = training_end - training_start
# prediction_time = prediction_end - prediction_start
# tuning_time = tuning_end - tuning_start

# # Output metrics
# print("\nRandom Forest Regressor Model Performance (After Hyper-Tuning):")
# print("\nTraining Set:")
# print(f"  - RMSE: {rmse_train:.4f}")
# print(f"  - MAE: {mae_train:.4f}")
# print(f"  - R^2 Score: {r2_train:.4f}")
# print(f"  - Training Time: {train_time:.4f} seconds")

# print("\nTesting Set:")
# print(f"  - RMSE: {rmse_test:.4f}")
# print(f"  - MAE: {mae_test:.4f}")
# print(f"  - R^2 Score: {r2_test:.4f}")
# print(f"  - Prediction Time: {prediction_time:.5f} seconds")

# print("\nHyperparameter Tuning Time:", f"{tuning_time:.4f} seconds")
