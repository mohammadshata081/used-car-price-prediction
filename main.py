from src.train import train_model
from src.evaluate import evaluate_model
import os

if __name__ == "__main__":
    DATA_PATH = 'car details v4.csv'
    MODELS_DIR = 'models'
    OUTPUTS_DIR = 'outputs'
    
    print("Starting Machine Learning Pipeline...")
    train_model(DATA_PATH, MODELS_DIR)
    print("-" * 30)
    evaluate_model(DATA_PATH, MODELS_DIR, OUTPUTS_DIR)
    print("Pipeline finished successfully!")
