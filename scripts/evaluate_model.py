import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model
from data_preprocessing import load_data

def evaluate_model(model, X_test, y_test):
    # Load data and model
    _, _, X_test, y_test = load_data()
    model = load_model('CNN_model.h5')

    # Predictions
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    # Calculate metrics (remove np.argmax for y_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix with correct labels
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, cbar=False, 
                xticklabels=['Predicted Normal', 'Predicted Pneumonia'], 
                yticklabels=['True Normal', 'True Pneumonia'],
                center=0)

    # Set label positions
    plt.gca().xaxis.tick_top()  # Move x-axis ticks to the top
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    plt.title('Confusion Matrix', pad=20)  # Add title with padding to avoid overlap

    plt.show()
