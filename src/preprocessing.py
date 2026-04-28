import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
    df = df.copy()
    
    # Extract numerical values from strings
    for col in ['Engine', 'Max Power', 'Max Torque']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
            
    # Drop 'Model' because it's too high cardinality
    if 'Model' in df.columns:
        df = df.drop(columns=['Model'])
        
    return df

def get_preprocessor(X):
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])
    return preprocessor
