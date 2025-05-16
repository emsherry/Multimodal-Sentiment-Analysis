# audio_cli.py
import argparse
import os
import numpy as np
from inference.predict import predict_emotion
from config import INV_EMOTION_LABELS

def analyze_audio(file_path):
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    # Predict
    emotion, probs = predict_emotion(file_path)

    # Get top-3 emotions
    sorted_indices = np.argsort(probs)[::-1]
    top3 = [(INV_EMOTION_LABELS[i], float(probs[i])) for i in sorted_indices[:3]]

    # Print results
    print({
        "emotion": top3[0][0],
        "confidence": top3[0][1],
        "top3": top3,
        "probabilities": list(map(float, probs))
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Sentiment Analyzer CLI")
    parser.add_argument("--audio", type=str, required=True, help="Path to .wav audio file")
    args = parser.parse_args()
    analyze_audio(args.audio)
