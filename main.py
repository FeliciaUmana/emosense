import io
import numpy as np
import librosa
import joblib
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
os.environ["PATH"] += r";C:\Users\HP\Downloads\ffmpeg-8.1.1-essentials_build (1)\ffmpeg-8.1.1-essentials_build\bin"

app = FastAPI(title='Speech Emotion Recognition API')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
model         = tf.keras.models.load_model("emotion_model.h5")
scaler        = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def extract_features(audio_bytes, n_mfcc=40):
    import subprocess, tempfile
    ffmpeg_path =  r"C:\Users\HP\Downloads\ffmpeg-8.1.1-essentials_build (1)\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"
    with tempfile.NamedTemporaryFile(suffix='.input', delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name
    tmp_out_path = tmp_in_path + '.wav'
    subprocess.run([ffmpeg_path, '-y', '-i', tmp_in_path, tmp_out_path], capture_output=True)
    
    audio, sr = librosa.load(tmp_out_path, res_type='kaiser_fast')
    os.remove(tmp_in_path)
    os.remove(tmp_out_path)
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    combined = np.hstack([
        np.mean(mfcc.T, axis=0),
        np.mean(librosa.feature.delta(mfcc).T, axis=0),
        np.mean(librosa.feature.delta(mfcc, order=2).T, axis=0),
        np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0),
        np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0),
        np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)])
    return combined
@app.get('/')
def root():
    return {'message': 'Speech Emotion API is running. Go to/docs to test.'}
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    features = extract_features(audio_bytes)
    scaled = scaler.transform([features]).reshape(1, -1, 1)
    predictions = model.predict(scaled, verbose=0)[0]
    emotion = label_encoder.inverse_transform([np.argmax(predictions)])[0]
    confidence = round(float(np.max(predictions))*100, 2)
    all_probs = {label_encoder.classes_[i]:round(float(predictions[i])*100, 2) for i in range(len(predictions))}
    return {'emotion': emotion, 'confidence':confidence, 'all_probabilities': all_probs}

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
