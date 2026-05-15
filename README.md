# Emotion Recognition from Speech API

Recognizes human emotions from speech audio using deep learning.

## Live API
https://emotion-recognition-api-pywn.onrender.com

## Emotions Detected
angry, calm, disgust, fearful, happy, neutral, sad, surprised

## Endpoints
- GET  /health  → check if model is loaded
- POST /predict → upload .wav file and get emotion prediction
- GET  /docs    → interactive API documentation

## Tech Stack
- Deep Learning: CNN + LSTM (TensorFlow/Keras)
- Feature Extraction: MFCC (librosa)
- Dataset: RAVDESS (1440 audio files, 8 emotions)
- API: FastAPI
- Deployment: Render