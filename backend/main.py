import os
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File
from inference.predict import predict_emotion
from config import INV_EMOTION_LABELS  # Make sure you have this

app = FastAPI()

UPLOAD_DIR = "backend/data/raw_audio/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Get prediction
    emotion, probs = predict_emotion(file_path)

    # Get top-3 predictions
    sorted_indices = np.argsort(probs)[::-1]
    top3 = [(INV_EMOTION_LABELS[i], float(probs[i])) for i in sorted_indices[:3]]

    return {
        "emotion": top3[0][0],
        "confidence": top3[0][1],
        "top3": top3,
        "probabilities": list(map(float, probs))
    }
