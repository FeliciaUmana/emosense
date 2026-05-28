-  # 🎙️ EmoSense — Speech Emotion Recognition

A deep learning system that detects human emotions from voice recordings using a CNN-BiLSTM model, deployed as a REST API with FastAPI.

---

## 🌐 Deployment

Deployed on **Render.com** as a free web service.

## Live API
https://emosense-zd8b.onrender.com

## Fontend live Test
[https://feliciaumana.github.io/emosense/](https://keen-narwhal-889fb8.netlify.app/)







## 🧠 Model Overview

The model is trained on 4 combined speech emotion datasets and classifies audio into 8 emotions:

`angry` · `happy` · `sad` · `fearful` · `neutral` · `surprised` · `fear` · `surprised`

### Architecture
- **CNN Block 1** — 64 filters, kernel size 3, BatchNorm, MaxPooling, Dropout
- **CNN Block 2** — 128 filters, kernel size 3, BatchNorm, MaxPooling, Dropout
- **BiLSTM Layer** — 128 units, bidirectional
- **Dense Layers** — 128 → 64 → output (softmax)

### Training Results
| Metric | Value |
|--------|-------|
| Training Accuracy | 75.74% |
| Validation Accuracy | 68.10% |
| Epochs | 105 (early stopping) |
| Optimizer | Adam (lr=0.0005) |

---

## 📦 Datasets Used

| Dataset | Emotions |
|---------|----------|
| CREMA-D | angry, disgust, fear, happy, neutral, sad |
| RAVDESS | neutral, calm, happy, sad, angry, fearful, disgust, surprised |
| SAVEE | angry, disgust, fearful, happy, neutral, sad, surprised |
| TESS | angry, disgust, fearful, happy, neutral, sad, surprised |

> **Note:** `calm` is merged into `neutral` and `disgust` into `angry` to reduce confusion between acoustically similar emotions.

---

## 🔧 Features Extracted

Each audio file is converted into a feature vector combining:
- **MFCC** (40 coefficients) — timbral texture
- **MFCC Delta** — velocity of change
- **MFCC Delta2** — acceleration of change
- **Chroma STFT** — pitch/harmony
- **Mel Spectrogram** — frequency energy
- **Spectral Contrast** — loudness variation

---

## 🚀 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| POST | `/predict` | Upload audio → get emotion |

### Example Response
```json
{
  "emotion": "happy",
  "confidence": 87.43,
  "all_probabilities": {
    "happy": 87.43,
    "neutral": 6.12,
    "angry": 3.21,
    "sad": 1.85,
    "fearful": 0.94,
    "surprised": 0.45
  }
}
```

---

## 🖥️ Local Setup

### Prerequisites
- Python 3.10+
- ffmpeg installed on your system

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn main:app --reload
```

## 📁 Project Structure

```
emosense/
├── main.py              # FastAPI application
├── index.html           # Frontend UI
├── emotion_model.h5     # Trained Keras model
├── scaler.pkl           # StandardScaler
├── label_encoder.pkl    # LabelEncoder
├── requirements.txt     # Dependencies
├── render.yaml          # Render deployment config
└── README.md
```


## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Model | TensorFlow / Keras |
| Feature Extraction | Librosa |
| API | FastAPI |
| Server | Uvicorn |
| Audio Conversion | ffmpeg |
| Deployment | Render.com |
| Frontend | HTML / CSS / JavaScript |

---

## 👤 Author

Built as a Speech Emotion Recognition project using deep learning on multiple datasets.
