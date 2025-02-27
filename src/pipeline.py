from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from outliers import OutlierReplaceWithMedian
from missing_value import MissingValueHandler
from feature_eng import FeatureEngineering
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from label_enco import CustomLabelEncoder

def build_pipeline(model):
    """
    Build a pipeline with the given model as the final step.
    If model is None, return only the preprocessing pipeline.
    
    Parameters:
        model: The machine learning model to be used in the pipeline.
        
    Returns:
        Pipeline: A scikit-learn pipeline.
    """
    # Column Transformer for Encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('label', CustomLabelEncoder(), ['sex']),
            ('label1', CustomLabelEncoder(), ['smoker']),
            ('one_hot', OneHotEncoder(drop='first', dtype=int), ['region']),
        ],
        remainder='passthrough'  # Keep other columns as they are
    )

    # Complete Preprocessing Pipeline
    preprocessing_pipeline = Pipeline(steps=[
        ('outlier_handling', OutlierReplaceWithMedian(threshold=1.5)),
        ('missing_value_handling', MissingValueHandler()),
        ('feature_engineering', FeatureEngineering(poly_degree=2)),
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler())
    ])
    
    # Return only the preprocessing pipeline if model is None
    if model is None:
        return preprocessing_pipeline
    
    # Otherwise, return full pipeline with model
    return Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('model', model)
    ])
