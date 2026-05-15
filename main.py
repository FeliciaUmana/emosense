from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import joblib
import tempfile
import os
from tensorflow.keras.models import load_model

app = FastAPI(
    title="Emotion Recognition from Speech API",
    description="Predicts human emotions from speech audio files",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessors
try:
    model = load_model("emotion_model.h5")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.get("/")
def home():
    return {
        "message": "Emotion Recognition API is running!",
        "endpoints": {
            "predict": "/predict → POST (upload audio file)",
            "health": "/health → GET",
            "docs": "/docs → GET"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "emotions": list(le.classes_)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Extract MFCC features
        audio, sample_rate = librosa.load(tmp_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)

        # Clean up temp file
        os.unlink(tmp_path)

        # Preprocess
        features = scaler.transform([mfccs_scaled])
        features = features.reshape(features.shape[0], features.shape[1], 1)

        # Predict
        prediction = model.predict(features)
        emotion_index = np.argmax(prediction[0])
        emotion = le.classes_[emotion_index]
        confidence = float(prediction[0][emotion_index])

        # All emotions with probabilities
        all_emotions = {
            le.classes_[i]: round(float(prediction[0][i]), 4)
            for i in range(len(le.classes_))
        }

        return {
            "predicted_emotion": emotion,
            "confidence": round(confidence, 4),
            "confidence_percent": f"{confidence * 100:.1f}%",
            "all_emotions": all_emotions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")