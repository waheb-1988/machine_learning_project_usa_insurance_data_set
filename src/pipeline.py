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
    
    Parameters:
        model: The machine learning model to be used in the pipeline.
        
    Returns:
        Pipeline: A scikit-learn pipeline.
    """
    # Column Transformer for Encoding
    print('model_name:',model,"#"*15)
    preprocessor = ColumnTransformer(
        transformers=[
            ('label', CustomLabelEncoder(), ['sex']),
            ('label1', CustomLabelEncoder(), ['smoker']),
            ('one_hot', OneHotEncoder(drop='first', dtype=int), ['region']),
        ],
        remainder='passthrough'  # Keep other columns as they are
    )

    # Complete Pipeline
    pipeline = Pipeline(steps=[
        ('outlier_handling', OutlierReplaceWithMedian(threshold=3)),
        ('missing_value_handling', MissingValueHandler()),
        ('feature_engineering', FeatureEngineering(poly_degree=2)),
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('model', model)  # Model passed as parameter
    ])
    
    return pipeline


            
            
