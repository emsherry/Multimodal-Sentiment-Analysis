# backend/inference/predict.py

import numpy as np
import tensorflow as tf
from config import MODEL_PATH, INV_EMOTION_LABELS, SPECTROGRAM_SHAPE
from preprocess.generate_spectrograms import generate_mel_spectrogram

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def predict_emotion(audio_path):
    spect = generate_mel_spectrogram(audio_path)
    spect = spect[:, :SPECTROGRAM_SHAPE[1]]
    spect = spect[np.newaxis, ..., np.newaxis]
    probs = model.predict(spect)[0]
    pred = INV_EMOTION_LABELS[np.argmax(probs)]
    return pred, probs
