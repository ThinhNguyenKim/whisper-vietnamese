from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from werkzeug.utils import secure_filename
import os
import whisper
import torch
from datetime import datetime

from utils import hf_to_whisper_states

from enum import Enum

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load HF Model
hf_state_dict = torch.load('small_vi_02.bin', map_location=torch.device('cpu'))    # *.bin file

# Rename layers
for key in list(hf_state_dict.keys())[:]:
    new_key = hf_to_whisper_states(key)
    hf_state_dict[new_key] = hf_state_dict.pop(key)

# Init Whisper Model and replace model weights
model = whisper.load_model('small')
model.load_state_dict(hf_state_dict)


class language(str, Enum):
    en = 'English'
    vi = 'Vietnamese'


@app.post("/Whisper/")
async def speech_to_text(file: UploadFile, language: language):
    audio_path = file.filename

    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    options = whisper.DecodingOptions(
    language=language, without_timestamps=True, fp16=torch.cuda.is_available()
)
    
    result = whisper.decode(model, mel, options)

    return {"transcribe": result.text}