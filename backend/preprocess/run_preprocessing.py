# backend/preprocess/run_preprocessing.py

import os
import numpy as np
from preprocess.generate_spectrograms import generate_mel_spectrogram, extract_emotion

RAW_DIR = "data/raw_audio/"
OUT_DIR = "data/spectrograms/"
os.makedirs(OUT_DIR, exist_ok=True)

for file in os.listdir(RAW_DIR):
    if file.endswith(".wav"):
        emotion = extract_emotion(file)
        if emotion:
            mel = generate_mel_spectrogram(os.path.join(RAW_DIR, file))
            out_file = f"{emotion}_{file.replace('.wav', '.npy')}"
            np.save(os.path.join(OUT_DIR, out_file), mel)
