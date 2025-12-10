import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cough_classifier_model.h5")
LOG_FILE = os.path.join(BASE_DIR, "history_log.csv")

# Audio Hyperparameters
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MAX_PAD_LENGTH = 256

# App Constraints
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB Limit
RATE_LIMIT_DURATION = 60  # seconds
RATE_LIMIT_REQUESTS = 10  # requests per duration

# Model Classes
CLASS_NAMES = ['dry', 'non-cough', 'wet']

# CORS Origins
ORIGINS = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8081",
    "http://127.0.0.1:8080",
    "http://localhost:5173",
    "https://kofkof.vercel.app"
]
