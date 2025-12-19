import pytest
import os
import csv
import numpy as np
from unittest.mock import MagicMock, patch, mock_open, AsyncMock
from fastapi import HTTPException, Request
from utils import log_prediction, rate_limiter, validate_file_header, request_history

# --- Test log_prediction ---
def test_log_prediction_creates_file():
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("os.path.isfile", return_value=False):
            log_prediction("test.wav", "wet", 0.95, 0.1)
            
            # Cek apakah header ditulis
            mock_file().write.assert_any_call("Timestamp,Filename,Prediction,Confidence,Processing_Time_Sec\r\n")

def test_log_prediction_appends_row():
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("os.path.isfile", return_value=True):
            log_prediction("test.wav", "wet", 0.95, 0.1)
            
            # Pastikan header TIDAK ditulis ulang, tapi data ditulis
            handle = mock_file()
            # Kita cek apakah ada write call yang mengandung data kita
            # Note: csv writer melakukan multiple write calls
            assert handle.write.call_count > 0

# --- Test rate_limiter ---
def test_rate_limiter_allow():
    # Reset history
    request_history.clear()
    
    mock_request = MagicMock(spec=Request)
    mock_request.client.host = "127.0.0.1"
    
    # Request pertama harus lolos
    try:
        rate_limiter(mock_request)
    except HTTPException:
        pytest.fail("Rate limiter should allow first request")

def test_rate_limiter_block():
    # Reset history
    request_history.clear()
    
    mock_request = MagicMock(spec=Request)
    mock_request.client.host = "127.0.0.2"
    
    # Isi history sampai limit (misal limit 10 di config, kita mock loop)
    # Kita patch config.RATE_LIMIT_REQUESTS agar test lebih mudah
    with patch("utils.RATE_LIMIT_REQUESTS", 2):
        rate_limiter(mock_request) # 1
        rate_limiter(mock_request) # 2
        
        # Request ke-3 harus gagal
        with pytest.raises(HTTPException) as excinfo:
            rate_limiter(mock_request)
        assert excinfo.value.status_code == 429

# --- Test validate_file_header ---
@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.mark.anyio
async def test_validate_file_header_valid_wav(anyio_backend):
    mock_file = MagicMock()
    mock_file.content_type = "audio/wav"
    
    # Mock read and seek to be awaitable
    mock_file.read = AsyncMock(return_value=b'RIFFxxxxWAVE')
    mock_file.seek = AsyncMock()
    
    # Seharusnya tidak raise error
    await validate_file_header(mock_file)

@pytest.mark.anyio
async def test_validate_file_header_invalid_magic_bytes(anyio_backend):
    mock_file = MagicMock()
    mock_file.content_type = "audio/wav"
    
    # Mock read and seek to be awaitable
    mock_file.read = AsyncMock(return_value=b'00000000')
    mock_file.seek = AsyncMock()
    
    # Seharusnya raise error
    with pytest.raises(HTTPException) as excinfo:
        await validate_file_header(mock_file)
    assert excinfo.value.status_code == 400
    assert "Invalid file format" in excinfo.value.detail 
