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

@patch('main.model', None)
def test_predict_model_not_loaded():
    files = {'file': ('test.wav', b'fake_audio_bytes', 'audio/wav')}
    response = client.post("/predict", files=files)
    
    # Expecting error because model is None
    assert response.json() == {"error": "Model not loaded."}

@patch('main.validate_file_header', new_callable=AsyncMock)
def test_predict_file_too_large(mock_validate):
    mock_validate.return_value = None
    
    # Create a large dummy content (> 5MB)
    # We mock the file read to simulate a large file without actually creating one in memory if possible,
    # but TestClient handles files differently. 
    # Instead of creating a 5MB string, let's patch the MAX_FILE_SIZE in main.py
    
    with patch('main.MAX_FILE_SIZE', 10): # Set limit to 10 bytes
        files = {'file': ('large.wav', b'123456789012345', 'audio/wav')}
        response = client.post("/predict", files=files)
        
        assert response.status_code == 413
        assert "File too large" in response.json()['detail']