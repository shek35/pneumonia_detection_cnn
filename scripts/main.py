import numpy as np
from tensorflow.keras.models import load_model
from data_preprocessing import load_data
from model_building import build_model
from train_model import train_model
from evaluate_model import evaluate_model

def main():
    # Step 1: Data Preprocessing
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test = load_data()
    
    # Step 2: Model Building
    print("Building model...")
    model = build_model(input_shape=X_train[0].shape)
    
    # Step 3: Model Training
    print("Training model...")
    model, history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Step 4: Model Evaluation
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
