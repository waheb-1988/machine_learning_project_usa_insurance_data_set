import numpy as np
import time
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pipeline import build_pipeline

def hyperparameter_tuning(model, param_grid, df, target_column="charges", test_size=0.25, random_state=20):
    """
    Hyperparameter tuning and model evaluation function.
    
    Parameters:
        model: A scikit-learn compatible model for hyperparameter tuning
        param_grid (dict): The hyperparameter grid for GridSearchCV
        df (pd.DataFrame): The dataset including features and target column
        target_column (str): The target variable for prediction
        test_size (float): Test set size as a fraction
        random_state (int): Seed for reproducibility
        
    Returns:
        None: Prints model performance and best parameters
    """
    # Data Preparation
    X = df.drop(columns=target_column)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Split Pipeline for Efficient Transformation
    preprocessor = build_pipeline(None)  # Get only the preprocessing steps
    
    # Preprocess once
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Grid Search
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    
    # Timing training
    training_start = time.perf_counter()
    grid_search.fit(X_train_transformed, y_train)
    training_end = time.perf_counter()
    
    best_model = grid_search.best_estimator_
    
    # Timing prediction
    prediction_start = time.perf_counter()
    y_pred_train = best_model.predict(X_train_transformed)
    y_pred_test = best_model.predict(X_test_transformed)
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
    print(f"\n{model.__class__.__name__.upper()} Model Performance:")
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

