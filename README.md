# Emotion Recognition from Speech API

Recognizes human emotions from speech audio using deep learning.

## Live API
https://emosense-zd8b.onrender.com

## Fontend live Test
https://feliciaumana.github.io/emosense/

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

## What you will see
-  A trained CNN-BiLSTM emotion recognition model
-  A live FastAPI backend on Render
-  A beautiful frontend UI
