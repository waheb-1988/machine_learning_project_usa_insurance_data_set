from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def train_and_evaluate(preprocessor, model_pipeline, df, target_column='charges', test_size=0.25, random_state=20):
    """
    Train and evaluate a model pipeline.
    
    Parameters:
        preprocessor (Pipeline): The preprocessing part of the pipeline.
        model_pipeline (Pipeline): The model part of the pipeline.
        df (pd.DataFrame): The input dataframe containing features and target.
        target_column (str): The target column name for prediction.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.
        
    Returns:
        None: Prints evaluation metrics for training and testing sets.
    """
    # Split data
    X = df.drop(columns=target_column)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Fit the preprocessor and transform data once
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Fit the model
    model_pipeline.fit(X_train_transformed, y_train)
    
    # Make predictions
    y_pred_train = model_pipeline.predict(X_train_transformed)
    y_pred_test = model_pipeline.predict(X_test_transformed)
    
    # Calculate KPIs
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Output metrics
    print("\nModel Performance:")
    print("\nTraining Set:")
    print(f"  - RMSE: {rmse_train:.4f}")
    print(f"  - MAE: {mae_train:.4f}")
    print(f"  - R^2 Score: {r2_train:.4f}")
    
    print("\nTesting Set:")
    print(f"  - RMSE: {rmse_test:.4f}")
    print(f"  - MAE: {mae_test:.4f}")
    print(f"  - R^2 Score: {r2_test:.4f}")