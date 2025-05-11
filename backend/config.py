# backend/config.py

EMOTION_LABELS = {
    'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3,
    'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7
}
INV_EMOTION_LABELS = {v: k for k, v in EMOTION_LABELS.items()}

SPECTROGRAM_SHAPE = (128, 256)
MODEL_PATH = "model/audio_emotion_model.h5"
