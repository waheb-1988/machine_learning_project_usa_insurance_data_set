from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

# Custom Transformer for Label Encoding
class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}
        
    def fit(self, X, y=None):
        for column in X.columns:
            le = LabelEncoder()
            le.fit(X[column])
            self.label_encoders[column] = le
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for column, le in self.label_encoders.items():
            X_transformed[column] = le.transform(X[column])
        return X_transformed

# Use the custom transformer in ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('label', CustomLabelEncoder(), ['sex'])  # Custom label encoding for 'sex'
    ],
    remainder='passthrough'  # Keep other columns unchanged
)
