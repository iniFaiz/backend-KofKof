import os
import numpy as np
import librosa
import csv
import time
from datetime import datetime
from collections import defaultdict
from fastapi import HTTPException, Request
from config import *

# Logging Function
def log_prediction(filename, prediction, confidence, processing_time):
    try:
        file_exists = os.path.isfile(LOG_FILE)
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Timestamp', 'Filename', 'Prediction', 'Confidence', 'Processing_Time_Sec'])
            
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                filename,
                prediction,
                f"{confidence:.4f}",
                f"{processing_time:.4f}"
            ])
    except Exception as e:
        print(f"Error writing log: {e}")

# Preprocessing Function
def preprocess_audio(audio_path: str):
    try:
        # Load using librosa
        y, sr = librosa.load(audio_path, sr=SR, mono=True, duration=10.0)
            
        # Mel-spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        log_S = librosa.power_to_db(S, ref=np.max)
        
        # Padding
        if log_S.shape[1] > MAX_PAD_LENGTH:
            log_S = log_S[:, :MAX_PAD_LENGTH]
        elif log_S.shape[1] < MAX_PAD_LENGTH:
            pad_width = MAX_PAD_LENGTH - log_S.shape[1]
            log_S = np.pad(log_S, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        # Normalization
        log_S = (log_S - np.mean(log_S)) / np.std(log_S)
        
        # Reshape
        log_S = log_S.reshape(1, N_MELS, MAX_PAD_LENGTH, 1)
        
        return log_S
    
    except Exception as e:
        print(f"Error in preprocessing audio: {e}")
        return None

# Rate Limiter
request_history = defaultdict(list)

def rate_limiter(request: Request):
    client_ip = request.client.host
    now = time.time()
    
    # Clean old requests
    request_history[client_ip] = [t for t in request_history[client_ip] if now - t < RATE_LIMIT_DURATION]
    
    if len(request_history[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
    
    request_history[client_ip].append(now)

# File Validation
async def validate_file_header(file):
    # Validasi Content Type
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    # Validasi Magic Bytes (Signature Check)
    header = await file.read(2048)
    await file.seek(0) # Reset cursor
    
    is_valid_format = False
    
    # WebM
    if header.startswith(b'\x1A\x45\xDF\xA3'): 
        is_valid_format = True
    # WAV (RIFF)
    if header.startswith(b'RIFF'): 
        is_valid_format = True
    # OGG (OggS)
    elif header.startswith(b'OggS'): 
        is_valid_format = True
    # FLAC (fLaC)
    elif header.startswith(b'fLaC'): 
        is_valid_format = True
    # MP3 (ID3 atau Sync Frame)
    elif header.startswith(b'ID3') or header.startswith(b'\xff\xfb') or header.startswith(b'\xff\xf3') or header.startswith(b'\xff\xf2'): 
        is_valid_format = True
    # M4A/AAC (ftyp di offset 4)
    elif len(header) > 8 and header[4:8] == b'ftyp': 
        is_valid_format = True
        
    if not is_valid_format:
         raise HTTPException(status_code=400, detail="Invalid file format. Magic bytes do not match supported audio formats (WAV, MP3, OGG, FLAC, M4A).")
