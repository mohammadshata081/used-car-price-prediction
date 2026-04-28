import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor
import joblib
import os
from src.preprocessing import load_data, clean_data, get_preprocessor

def train_model(data_path, models_dir):
    print("Loading data...")
    df = load_data(data_path)
    
    print("Cleaning data...")
    df = clean_data(df)
    
    # Drop rows where target is missing if any
    df = df.dropna(subset=['Price'])
    
    # Target variable (will be log transformed internally by the model pipeline)
    y = df['Price']
    X = df.drop(columns=['Price'])
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)
    
    print("Preprocessing data and configuring pipeline...")
    preprocessor = get_preprocessor(X_train)
    
    # Base XGBoost model
    xgb_base = XGBRegressor(random_state=42)
    
    # Wrap model to automatically log-transform the target during training and exponentiate during prediction
    target_model = TransformedTargetRegressor(regressor=xgb_base, func=np.log, inverse_func=np.exp)
    
    # Combine preprocessor and model into a single pipeline to prevent data leakage during Cross Validation
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', target_model)
    ])
    
    print("Performing 5-Fold Cross Validation to find optimal hyperparameters...")
    param_distributions = {
        'model__regressor__learning_rate': [0.05, 0.1, 0.2, 0.3],
        'model__regressor__n_estimators': [300, 500, 700],
        'model__regressor__max_depth': [3, 4, 5, 6, 7],
        'model__regressor__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'model__regressor__min_child_weight': [1, 3, 5]
    }
    
    # Using RandomizedSearchCV to efficiently search the hyperparameter space with 5-fold CV
    search = RandomizedSearchCV(
        full_pipeline, 
        param_distributions=param_distributions, 
        n_iter=40, 
        cv=5, 
        scoring='r2', 
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    
    print(f"Best CV R2 Score: {search.best_score_:.4f}")
    print(f"Best Parameters: {search.best_params_}")
    
    best_pipeline = search.best_estimator_
    
    print("Saving artifacts...")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_pipeline.named_steps['preprocessor'], os.path.join(models_dir, 'preprocessor.joblib'))
    joblib.dump(best_pipeline.named_steps['model'], os.path.join(models_dir, 'xgb_model.joblib'))
    
    print("Training complete!")

if __name__ == "__main__":
    train_model('car details v4.csv', 'models')
