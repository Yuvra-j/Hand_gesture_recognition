import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

gestures = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']

def load_gestures(base_dir):
    X = []
    y = []
    for subject in range(10):
        subject_dir = os.path.join(base_dir, f'{subject:02d}')
        for i, gesture in enumerate(gestures):
            folder = os.path.join(subject_dir, f'{i+1:02d}_{gesture}')
            for file in os.listdir(folder):
                if file.endswith('.png'):
                    img_path = os.path.join(folder, file)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (128, 128)) / 255.0
                    X.append(img)
                    y.append(i)
    return np.array(X), np.array(y)

def prepare_data(base_dir):
    X, y = load_gestures(base_dir)
    y = to_categorical(y, num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    base_dir = "C:/Users/Yuvraj/OneDrive/Desktop/dataset/leapGestRecog"
    X_train, X_test, y_train, y_test = prepare_data(base_dir)
    print(f'Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}')