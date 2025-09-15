Hand Gesture Recognition Model:
A deep learning model to classify 10 hand gestures using a Convolutional Neural Network (CNN). Enhanced with MediaPipe for hand detection.

Features
1. 100% test accuracy on "LeapGestRecog" dataset(easily available on kaggle)
2. Real-time gesture recognition with MediaPipe hand tracking.
3. Saved model for reuse.

Installation
1. Clone the repository: `git clone https://github.com/Yuvra-j/Hand_gesture_recognition.git`
2. Install dependencies: `pip install -r requirements.txt`

Usage
1. Train the model: `python train_model.py`
2. Predict in real-time: `python main.py` 

Results
1. Accuracy: 100% on test set aand around 98% accuracy on validation set.
2. MediaPipe improves hand focus in real time and improves overall performance.

Limitations
1. May need retraining for diverse lighting.
2. Static gestures only.

Future Work
1. Dynamic gesture support with LSTM.
2. Test on varied datasets.
