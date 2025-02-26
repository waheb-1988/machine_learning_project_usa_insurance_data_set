from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

class FeatureEngineering(BaseEstimator, TransformerMixin):
    
    def __init__(self, poly_degree=2, include_bias=False, drop_first=True):
        self.poly_degree = poly_degree
        self.include_bias = include_bias
        self.drop_first = drop_first
        self.poly = PolynomialFeatures(degree=self.poly_degree, include_bias=self.include_bias)
        self.poly_columns = []
        
    def fit(self, X, y=None):
        # Fit polynomial features
        self.poly.fit(X[['age', 'bmi']])
        self.poly_columns = self.poly.get_feature_names_out(['age', 'bmi'])
        return self
    
    def transform(self, x):
        X = x.copy()
        # 1. Feature Interaction
        X['bmi_age'] = X['bmi'] * X['age']
        # 3. Polynomial Features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(X[['age', 'bmi']])
        poly_df = pd.DataFrame(poly_features, columns=['age', 'bmi', 'age^2', 'bmi^2', 'age*bmi'])
        # Reset the index of both DataFrames
        X.reset_index(drop=True, inplace=True)
        poly_df.reset_index(drop=True, inplace=True)
       
        X= pd.concat([X, poly_df], axis=1)
        # Remove duplicate columns (if any)
        X = X.loc[:, ~X.columns.duplicated()]
      
        # 4. Binning
        X['age_bin'] = pd.cut(X['age'], bins=[0, 20, 30, 40, 50, 100], labels=[1, 2, 3, 4, 5])
        X['bmi_bin'] = pd.cut(X['bmi'], bins=[0, 18.5, 24.9, 29.9, 100], labels=[1, 2, 3, 4])
        
        # 5. Creating New Features
        X['bmi_category'] = pd.cut(X['bmi'], bins=[0, 18.5, 24.9, 29.9, 100], 
                                   labels=['underweight', 'normal', 'overweight', 'obese'])
        
        # 6. One-Hot Encoding
        X = pd.get_dummies(X, columns=['age_bin', 'bmi_bin', 'bmi_category'], 
                           drop_first=self.drop_first, dtype=int)
        
        # 7. Remove duplicate columns again (just in case)
        X = X.loc[:, ~X.columns.duplicated()]
       
        return X

      