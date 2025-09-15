import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.model import create_model
from data.load_data import prepare_data

def train_model():
    X_train, X_test, y_train, y_test = prepare_data("C:/Users/Yuvraj/gesture_model.h5")  

    datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
    )
    datagen.fit(X_train)

    model = create_model()
    for epoch in range(20):
        print(f"\nEpoch {epoch + 1}/20")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            epochs=1,
            validation_data=(X_test, y_test),
            verbose=1
        )
        model.save(f'gesture_model_epoch{epoch + 1}.h5')
    model.save('gesture_model.h5')
    print("Training completed and model saved")

if __name__ == "__main__":
    train_model()