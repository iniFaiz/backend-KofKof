import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
import sys

# Mock tensorflow modules to avoid loading heavy libraries during tests
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.keras'] = MagicMock()
sys.modules['tensorflow.keras.models'] = MagicMock()

# Now we can import main
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "Cough Classifier API is running."}

@patch('main.validate_file_header', new_callable=AsyncMock)
@patch('main.preprocess_audio')
@patch('main.model')
def test_predict_cough_success(mock_model, mock_preprocess, mock_validate):
    # Setup Mocks
    mock_validate.return_value = None # Validation passes
    
    # Mock preprocessing to return a dummy spectrogram (1, 128, 256, 1)
    mock_preprocess.return_value = np.zeros((1, 128, 256, 1))
    
    # Mock model prediction (returns numpy array)
    # Prediction: [Dry, Non-Cough, Wet] -> Let's predict 'Wet' (index 2)
    mock_model.predict.return_value = np.array([[0.05, 0.05, 0.9]])
    
    # Create dummy file
    files = {'file': ('test.wav', b'fake_audio_bytes', 'audio/wav')}
    
    # Make Request
    response = client.post("/predict", files=files)
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data['filename'] == 'test.wav'
    assert data['prediction'] == 'wet'
    assert "90.00%" in data['confidence']

def test_predict_invalid_content_type():
    # Test with text file
    files = {'file': ('test.txt', b'text content', 'text/plain')}
    
    # We expect 400 Bad Request because of content-type check
    # Note: We don't mock validate_file_header here to test the real logic of content-type check
    # But validate_file_header is async, and TestClient handles async endpoints.
    # However, we need to make sure validate_file_header is not mocked if we want to test its logic.
    # But since we imported 'validate_file_header' in main.py, we can just let it run.
    # The content-type check is the first thing in validate_file_header.
    
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "File must be an audio file" in response.json()['detail']

@patch('main.validate_file_header', new_callable=AsyncMock)
@patch('main.preprocess_audio')
@patch('main.model')
def test_predict_preprocessing_failure(mock_model, mock_preprocess, mock_validate):
    mock_validate.return_value = None
    
    # Simulate preprocessing failure (returns None)
    mock_preprocess.return_value = None
    
    files = {'file': ('test.wav', b'fake_audio_bytes', 'audio/wav')}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    assert "Error" in response.json()['error']
