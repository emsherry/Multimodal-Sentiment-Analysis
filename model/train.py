# backend/model/train.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from config import EMOTION_LABELS, SPECTROGRAM_SHAPE, MODEL_PATH
from model.cnn_lstm import build_cnn_lstm

def load_data(directory=None):
    # Use correct full path from root
    if directory is None:
        directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'spectrograms')
        directory = os.path.abspath(directory)

    X, y = [], []
    for file in os.listdir(directory):
        if file.endswith(".npy"):
            data = np.load(os.path.join(directory, file))
            label = file.split("_")[0]
            if label in EMOTION_LABELS:
                data = data[:, :SPECTROGRAM_SHAPE[1]]
                data = data[..., np.newaxis]
                X.append(data)
                y.append(EMOTION_LABELS[label])
    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=len(EMOTION_LABELS))
    return X, y

def train():
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    model = build_cnn_lstm()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)
    model.save(MODEL_PATH)

if __name__ == "__main__":
    train()
