# backend/gradio_app.py

import gradio as gr
import os
import numpy as np
from inference.predict import predict_emotion
from config import INV_EMOTION_LABELS
import matplotlib.pyplot as plt

# Emoji Mapping (you can customize these)
EMOJI_MAP = {
    "angry": "😠",
    "happy": "😄",
    "sad": "😢",
    "neutral": "😐",
    "fear": "😨",
    "disgust": "🤢",
    "surprise": "😲",
    "calm": "😌",
}

def analyze_audio_gradio(file):
    emotion, probs = predict_emotion(file)
    sorted_indices = np.argsort(probs)[::-1]
    top3 = [(INV_EMOTION_LABELS[i], float(probs[i])) for i in sorted_indices[:3]]
    top_emotion = top3[0][0]
    emoji = EMOJI_MAP.get(top_emotion, "")

    # Create chart for top 3
    labels = [label for label, _ in top3]
    values = [prob for _, prob in top3]

    fig, ax = plt.subplots()
    ax.bar(labels, values, color="skyblue")
    ax.set_ylim([0, 1])
    ax.set_title("Top 3 Emotions")

    return f"{top_emotion} {emoji}", fig

demo = gr.Interface(
    fn=analyze_audio_gradio,
    inputs=gr.Audio(type="filepath", label="Upload Audio (.wav only)"),
    outputs=["text", "plot"],
    title="🎧 Audio Emotion Classifier",
    description="Upload a .wav file and get emotion prediction with confidence chart"
)

if __name__ == "__main__":
    demo.launch()
