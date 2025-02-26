
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np 
import pathlib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

class MissingValueHandler(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.columns_touched = {
            'numerical': [],
            'categorical': []
        }
        self.medians = {}
        self.modes = {}

    def fit(self, X, y=None):
        # Store medians for numerical columns
        for col in X.select_dtypes(include=["float64", "int64"]).columns:
            self.medians[col] = X[col].median()
        
        # Store modes for categorical columns
        for col in X.select_dtypes(include=["object", "category"]).columns:
            self.modes[col] = X[col].mode()[0]
        
        return self
    
    def transform(self, x):
        X = x.copy()
        self.columns_touched = {
            'numerical': [],
            'categorical': []
        }  # Reset for each transformation
        
        # Handle missing values for numerical columns
        for col in X.select_dtypes(include=["float64", "int64"]).columns:
            if X[col].isna().any():
                X[col].fillna(self.medians[col], inplace=True)
                self.columns_touched['numerical'].append(col)
        
        # Handle missing values for categorical columns
        for col in X.select_dtypes(include=["object", "category"]).columns:
            if X[col].isna().any():
                X[col].fillna(self.modes[col], inplace=True)
                self.columns_touched['categorical'].append(col)
        
        # Print the names of columns where missing values were handled
        if self.columns_touched['numerical']:
            print(f"Missing values handled in numerical columns: {', '.join(self.columns_touched['numerical'])}")
        if self.columns_touched['categorical']:
            print(f"Missing values handled in categorical columns: {', '.join(self.columns_touched['categorical'])}")
        
        if not self.columns_touched['numerical'] and not self.columns_touched['categorical']:
            print("No missing values detected.")
        
        return X
    
      
            
  