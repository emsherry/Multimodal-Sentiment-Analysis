import os
import shutil
from fastapi import FastAPI, UploadFile, File
from inference.predict import predict_emotion

app = FastAPI()

UPLOAD_DIR = "backend/data/raw_audio/"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # ✅ Make sure it exists

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    emotion, probs = predict_emotion(file_path)
    onfidence = float(max(probs))
    return {
        "emotion": emotion,
        "confidence": confidence,
        "probabilities": list(map(float, probs))
    }

