import pandas as pd
from data_preparation import load_and_preprocess_data
from model_training import train_model

def main():
    # Path to the dataset
    file_path = '/mnt/data/cleaned_calls.csv'  # Adjust the file path accordingly
    
    # Load and preprocess the data
    X, y, preprocessor = load_and_preprocess_data(file_path)
    
    # Train the model and evaluate its performance
    train_model(X, y, preprocessor)

if __name__ == "__main__":
    main()
