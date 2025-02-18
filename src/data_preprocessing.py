import pandas as pd
import os
import pathlib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
import numpy as np

class Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, missing_value_strategy='default', outlier_method='IQR', verbose=True):
        self.missing_value_strategy = missing_value_strategy
        self.outlier_method = outlier_method
        self.verbose = verbose
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()

        # Apply each preprocessing function
        #df = self.handle_missing_values(df)
        print('########11########')
        print(df)
        df = self.remove_duplicates(df)
        print('########22########')
        print(df)
        df = self.handle_outliers(df)
        print('########33########')
        print(df)
        df = self.label_encode_columns(df)
        print('########22########')
        print(df)
        df = pd.get_dummies(df, columns=['region'], drop_first=True, dtype=int)
       
        df['bmi_age'] = df['bmi'] * df['age']
        
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(df[['age', 'bmi']])
        poly_df = pd.DataFrame(poly_features, columns=['age', 'bmi', 'age^2', 'bmi^2', 'age*bmi'])
        df = pd.concat([df, poly_df], axis=1)
        # Remove duplicate columns (if any)
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.iloc[:-1]
        df['age_bin'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 100], labels=[1, 2, 3, 4, 5])
        df['bmi_bin'] = pd.cut(df['bmi'], bins=[0, 18.5, 24.9, 29.9, 100], labels=[1, 2, 3, 4])

        df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['underweight', 'normal', 'overweight', 'obese'])
        df = pd.get_dummies(df, columns=['age_bin', 'bmi_bin', 'bmi_category'], drop_first=True, dtype=int)

        df = self.add_stat_columns(df, ["age", "bmi", "bmi_age", "age^2", "bmi^2", "age*bmi"])
        
        return df

    def handle_missing_values(self, df, strategy='default', custom_value=None):
        df_imputed = df.copy()
        
        # Handling Numerical Variables
        numeric_cols = df_imputed.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df_imputed[col].isnull().sum() > 0:
                if strategy == 'mean' or (strategy == 'default' and df_imputed[col].dtype in ['float64', 'int64']):
                    df_imputed[col].fillna(df_imputed[col].mean(), inplace=True)
                elif strategy == 'median':
                    df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
                elif strategy == 'mode':
                    df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)
                elif strategy == 'custom' and custom_value is not None:
                    df_imputed[col].fillna(custom_value, inplace=True)
        
        # Handling Categorical Variables
        categorical_cols = df_imputed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_imputed[col].isnull().sum() > 0:
                if strategy == 'mode' or (strategy == 'default' and df_imputed[col].dtype == 'object'):
                    df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)
                elif strategy == 'ffill':
                    df_imputed[col].fillna(method='ffill', inplace=True)
                elif strategy == 'bfill':
                    df_imputed[col].fillna(method='bfill', inplace=True)
                elif strategy == 'custom' and custom_value is not None:
                    df_imputed[col].fillna(custom_value, inplace=True)
        
        return df_imputed
    
    def remove_duplicates(self, df, subset=None, keep='first', inplace=False):
        if keep not in ['first', 'last', 'none']:
            raise ValueError("keep must be one of 'first', 'last', or 'none'.")
        
        total_rows = len(df)
        duplicate_rows = df.duplicated(subset=subset, keep=False).sum()
        percentage_duplicates = (duplicate_rows / total_rows) * 100
        
        if duplicate_rows == 0:
            return df
        
        if keep == 'none':
            duplicated_mask = df.duplicated(subset=subset, keep=False)
            result = df[~duplicated_mask]
        else:
            result = df.drop_duplicates(subset=subset, keep=keep)
        
        if inplace:
            df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        else:
            return result
    
    def handle_outliers(self, df, numerical_columns=None, method='IQR', verbose=True):
        if numerical_columns is None:
            numerical_columns = df.select_dtypes(include=['number']).columns
        
        for col in numerical_columns:
            if method == 'IQR':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                
                if verbose:
                    print(f"Outliers handled in '{col}' using IQR.")
            
            elif method == 'Z-Score':
                mean = df[col].mean()
                std_dev = df[col].std()
                threshold = 3
                
                outliers = ((df[col] < (mean - threshold * std_dev)) | (df[col] > (mean + threshold * std_dev)))
                lower_bound = mean - threshold * std_dev
                upper_bound = mean + threshold * std_dev
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                
                if verbose:
                    print(f"Outliers handled in '{col}' using Z-Score.")
        
        return df 
    
    def label_encode_columns(self, df, columns=['sex', 'smoker'], verbose=True):
        for col in columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                
                if verbose:
                    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                    print(f"Label Encoding for '{col}': {mapping}")
            else:
                print(f"Column '{col}' not found in DataFrame.")
        
        return df
    def read_file(self, file_name: str) -> pd.DataFrame:
        try:
            dir_folder = pathlib.Path().cwd()
            file_path = dir_folder / "data"
            df = pd.read_csv(os.path.join(file_path, file_name))
            return df
        except FileNotFoundError:
            print(f"Error: The file at '{file_name}' was not found.")
            raise
        except Exception as e:
            print(f"An error occurred: {e}")

    def add_stat_columns(self, df, columns_to_scale):
        for col in columns_to_scale:
            if col in df.columns:
                mean_col = f"{col}_mean"
                sd_col = f"{col}_sd"
                min_col = f"{col}_min"
                max_col = f"{col}_max"
                
                df[mean_col] = df[col].mean()
                df[sd_col] = df[col].std()
                df[min_col] = df[col].min()
                df[max_col] = df[col].max()
            else:
                print(f"Warning: Column '{col}' not found in DataFrame.")
        
        return df


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split


# Initialize Preprocessing object
pc = Preprocessing(missing_value_strategy='mean', outlier_method='IQR')
df = pc.read_file(file_name='insurance.csv')

x= df.drop(columns='charges')
y= df['charges']
# This will show all the column names after preprocessing
x_t,x_te,y_t,y_te= train_test_split(x,y,test_size=.25,random_state=20)
# Create the pipeline with preprocessing steps
pipeline = Pipeline([
    ('preprocessing', Preprocessing(missing_value_strategy='mean', outlier_method='IQR')),
    ('scaler', StandardScaler())
    
])

# Assuming df is your DataFrame
x_t = pipeline.fit_transform(x_t)
x_te = pipeline.fit_transform(x_te)



# Initialize Linear Regression model

lr= LinearRegression()
# Timing the training process
training_start = time.perf_counter()
model = lr.fit(x_t, y_t)
training_end = time.perf_counter()

# Timing the prediction process
prediction_start = time.perf_counter()
y_pred_train = model.predict(x_t)
y_pred_test = model.predict(x_te)
prediction_end = time.perf_counter()

# Calculate KPIs
# Root Mean Squared Error (RMSE)
rmse_train = np.sqrt(mean_squared_error(y_t, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_te, y_pred_test))

# Mean Absolute Error (MAE)
mae_train = mean_absolute_error(y_t, y_pred_train)
mae_test = mean_absolute_error(y_te, y_pred_test)

# R-squared score
r2_train = r2_score(y_t, y_pred_train)
r2_test = r2_score(y_te, y_pred_test)

# Time metrics
train_time = training_end - training_start
prediction_time = prediction_end - prediction_start

# Output metrics
print("Linear Regression Model Performance:")
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


    