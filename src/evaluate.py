import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from src.preprocessing import load_data, clean_data

def evaluate_model(data_path, models_dir, outputs_dir):
    print("Loading data for evaluation...")
    df = load_data(data_path)
    df = clean_data(df)
    df = df.dropna(subset=['Price'])
    
    y = df['Price']
    X = df.drop(columns=['Price'])
    
    # Use same split as training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=40)
    
    print("Loading model and preprocessor...")
    preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.joblib'))
    model = joblib.load(os.path.join(models_dir, 'xgb_model.joblib'))
    
    print("Making predictions...")
    X_test_processed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_processed)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE:      {mse:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"MAE:      {mae:.4f}")
    
    # 1. Parity Plot (Log Scale for better visualization)
    os.makedirs(outputs_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='#1f77b4')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Actual Price (₹) - Log Scale Axis')
    plt.ylabel('Predicted Price (₹) - Log Scale Axis')
    plt.title('Actual vs Predicted Used Car Prices')
    
    # Format axes to show plain numbers with commas instead of scientific notation
    ax = plt.gca()
    formatter = ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'parity_plot.png'))
    plt.close()
    
    # 2. Residual Plot
    residuals = y_pred - y_test
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, color='#d62728')
    plt.axhline(0, color='black', linestyle='--', lw=2)
    plt.xlabel('Predicted Price (₹)')
    plt.ylabel('Residual Error (₹)')
    plt.title('Residuals vs Predicted Price')
    
    ax2 = plt.gca()
    ax2.xaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'residual_plot.png'))
    plt.close()
    
    # 3. Feature Importance Plot
    try:
        # Extract feature names from ColumnTransformer
        num_features = preprocessor.transformers_[0][2]
        cat_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
        cat_features = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])
        feature_names = list(num_features) + list(cat_features)
        
        xgb_model = model.regressor_
        importances = xgb_model.feature_importances_
        
        # Sort and plot top 15 features
        indices = np.argsort(importances)[::-1][:15]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), top_importances[::-1], align='center', color='#2ca02c')
        plt.yticks(range(len(indices)), top_features[::-1])
        plt.xlabel('Relative Importance')
        plt.title('Top 15 Feature Importances (XGBoost)')
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, 'feature_importance.png'))
        plt.close()
    except Exception as e:
        print("Could not generate feature importance plot:", e)
        
    print("Evaluation complete! Plots saved in", outputs_dir)

if __name__ == "__main__":
    evaluate_model('dataset/car details v4.csv', 'models', 'outputs')
