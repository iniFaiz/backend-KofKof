import os
import uvicorn
import numpy as np
import tempfile
import tensorflow as tf
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

# Import Modules
from config import *
from utils import log_prediction, preprocess_audio, rate_limiter, validate_file_header

# Initialize App
app = FastAPI(
    title="KofKof API",
    description="API for KofKof application",
    version="1.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model {MODEL_PATH} loaded successfully.")
except Exception as e:
    print(f"Error loading model {MODEL_PATH}: {e}")
    model = None

# Endpoints
@app.get("/")
def read_root():
    return {"status": "Cough Classifier API is running."}

@app.post("/predict", dependencies=[Depends(rate_limiter)])
async def predict_cough(file: UploadFile = File(...)):
    start_time = time.time()
    if not model:
        return {"error": "Model not loaded."}
    
    # 1. Validate File Header & Magic Bytes
    await validate_file_header(file)

    # 2. Save to Temp File (with Size Limit)
    suffix = os.path.splitext(file.filename)[1]
    if not suffix:
        suffix = ".tmp"
        
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file_size = 0
            while True:
                chunk = await file.read(1024 * 1024) # Read 1MB chunk
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    tmp.close()
                    os.remove(tmp.name)
                    raise HTTPException(status_code=413, detail="File too large. Maximum size is 5MB.")
                tmp.write(chunk)
            tmp_path = tmp.name
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

    try:
        # 3. Preprocessing (Non-blocking)
        spectrogram = await run_in_threadpool(preprocess_audio, tmp_path)
        
        if spectrogram is None:
            return {"error": "Error in audio preprocessing or file is corrupted."}
        
        # 4. Prediction (Non-blocking)
        prediction = await run_in_threadpool(model.predict, spectrogram)
        
        predicted_index = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(np.max(prediction))
        
        # 5. Logging
        processing_time = time.time() - start_time
        log_prediction(file.filename, predicted_class, confidence, processing_time)
        
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
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)