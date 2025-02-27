
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np 

class OutlierReplaceWithMedian(BaseEstimator, TransformerMixin):
    
    def __init__(self, threshold=3):
        self.threshold = threshold
        self.columns_touched = []

    def fit(self, X, y=None):
        return self
    
    def transform(self, x):
        X = x.copy()
        self.columns_touched = []  # Reset for each transformation
        
        for col in X.select_dtypes(include=["float64", "int64"]).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR

            # Check if any outliers are present
            outliers = (X[col] < lower_bound) | (X[col] > upper_bound)
            if outliers.any():
                # Replace outliers with the median
                median = X[col].median()
                X[col] = np.where(outliers, median, X[col])
                # Track the column name
                self.columns_touched.append(col)
        
        # Print the names of columns where outliers were replaced
        if self.columns_touched:
            print(f"Outliers replaced in columns: {', '.join(self.columns_touched)}")
        else:
            print("No outliers detected.")
        
        return X

    
            

            

# Outliers replace.............> Done
# Inpute missing vale.............> Done
# Processing steps.............> Done
# Onehot and Catencawdin.............> Done
# Pipeline............> Done
# Model............> Done
# save ............> Done
# hyper
# Save
# test............> Done
# Flask
# Fast api
# Streamlit
# Requiremet file 
# Env file 
# Chatbot

