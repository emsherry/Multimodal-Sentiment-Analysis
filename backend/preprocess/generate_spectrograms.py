# backend/preprocess/generate_spectrograms.py

import os
import numpy as np
import librosa
from config import SPECTROGRAM_SHAPE

def generate_mel_spectrogram(file_path, sr=22050, n_mels=128, max_len=256):
    y, sr = librosa.load(file_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    if log_mel.shape[1] < max_len:
        pad_width = max_len - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel = log_mel[:, :max_len]

    return log_mel

def extract_emotion(file_name):
    # RAVDESS filename: '03-01-05-01-02-01-12.wav'
    emo_code = int(file_name.split("-")[2])
    emotion_map = {
        1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
        5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
    }
    return emotion_map.get(emo_code)
