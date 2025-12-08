import os
import uvicorn
import numpy as np
import librosa
import soundfile as sf
import io
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Pengaturan & Hyperparameter
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MAX_PAD_LENGTH = 256
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cough_classifier_model.h5")
CLASS_NAMES = ['dry', 'non-cough', 'wet']

# CORS Configuration
app = FastAPI(
    title="KofKof API",
    description="API for KofKof application",
    version="1.0"
)

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8081",
    "http://127.0.0.1:8080",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], #CRUD enable all methods
    allow_headers=["*"],
)

# Load Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model {MODEL_PATH} loaded successfully.")
except Exception as e:
    print(f"Error loading model {MODEL_PATH}: {e}")
    model = None
    
# Preprocessing audio
def preprocess_audio(audio_bytes: bytes):
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes)) # load memory file
        
        # Jika stereo, convert ke mono
        if y.ndim > 1:
            y = y.T
            y = librosa.to_mono(y)
        
        # Resample ke SR
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            
        # Buat Mel-spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        log_S = librosa.power_to_db(S, ref=np.max)
        
        # Padding
        if log_S.shape[1] > MAX_PAD_LENGTH:
            log_S = log_S[:, :MAX_PAD_LENGTH]
        elif log_S.shape[1] < MAX_PAD_LENGTH:
            pad_width = MAX_PAD_LENGTH - log_S.shape[1]
            log_S = np.pad(log_S, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        # Normalisasi
        log_S = (log_S - np.mean(log_S)) / np.std(log_S)
        
        # Dimensi batch dan channel untuk model
        log_S = log_S.reshape(1, N_MELS, MAX_PAD_LENGTH, 1)
        
        return log_S
    
    except Exception as e:
        print(f"Error in preprocessing audio: {e}")
        return None
    
# Endpoint
@app.get("/")
def read_root():
    return {"status": "Cough Classifier API is running."}

@app.post("/predict")
async def predict_cough(file: UploadFile = File(...)):
    if not model:
        return {"error": "Model not loaded."}
    
    audio_bytes = await file.read()
    
    spectrogram = preprocess_audio(audio_bytes)
    
    if spectrogram is None:
        return {"error": "Error in audio preprocessing."}
    
    # Begin prediction
    prediction = model.predict(spectrogram)
    
    predicted_index = np.argmax(prediction)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(prediction))
    
    return {
        "filename": file.filename,
        "prediction": predicted_class,
        "confidence": f"{confidence * 100:.2f}%",
        "raw_scores": {
            CLASS_NAMES[0]: f"{prediction[0][0] * 100:.2f}%",
            CLASS_NAMES[1]: f"{prediction[0][1] * 100:.2f}%",
            CLASS_NAMES[2]: f"{prediction[0][2] * 100:.2f}%"
        }
    }
    
if __name__ == "__main__":
    print("Starting Cough Classifier API...")
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)