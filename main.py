import os
import uvicorn
import numpy as np
import librosa
import soundfile as sf
import io
import tempfile
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

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
def preprocess_audio(audio_path: str):
    try:
        # Load menggunakan librosa (batasi durasi misal 10 detik agar tidak OOM)
        y, sr = librosa.load(audio_path, sr=SR, mono=True, duration=10.0)
            
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
    
    # Validasi Content Type
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    # Simpan ke temp file dengan ekstensi yang benar
    suffix = os.path.splitext(file.filename)[1]
    if not suffix:
        suffix = ".tmp" # Default suffix if none provided
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Jalankan preprocessing di thread pool agar tidak blocking
        spectrogram = await run_in_threadpool(preprocess_audio, tmp_path)
        
        if spectrogram is None:
            return {"error": "Error in audio preprocessing or file is corrupted."}
        
        # Jalankan prediksi model (juga bisa blocking)
        prediction = await run_in_threadpool(model.predict, spectrogram)
        
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
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
if __name__ == "__main__":
    print("Starting Cough Classifier API...")
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)