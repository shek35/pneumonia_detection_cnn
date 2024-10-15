import numpy as np
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam
from data_preprocessing import load_data
from model_building import build_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(model, X_train, y_train, X_test, y_test):
    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # One-hot encode labels
    y_train = utils.to_categorical(y_train, num_classes=2)
    y_test = utils.to_categorical(y_test, num_classes=2)

    print("Data loaded and labels one-hot encoded")

    # Build the model
    model = build_model(X_train[0].shape)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow(X_train, y_train, batch_size=16)
    test_generator = datagen.flow(X_test, y_test, batch_size=16)

    # Train the model
    checkpoint = ModelCheckpoint('model_checkpoint.keras', save_best_only=True)

    history = model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_test, y_test),
        batch_size=16,
        callbacks=[checkpoint]
    )
    
    # Save model
    model.save('CNN_model.h5')
    print("Model saved as CNN_model.h5")

    return model, history