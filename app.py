from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import os
from datetime import datetime

import whisper
import torch
from config import Config
from model import WhisperModelModule

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


# Load model small
## Load HF Model
hf_state_dict = torch.load('small_vi_02.bin', map_location=torch.device(device))    # *.bin file

## Rename layers
for key in list(hf_state_dict.keys())[:]:
    new_key = hf_to_whisper_states(key)
    hf_state_dict[new_key] = hf_state_dict.pop(key)

## Init Whisper Model and replace model weights
model = whisper.load_model('small')
model.load_state_dict(hf_state_dict)


# Load model base
config = Config()
config.checkpoint_path = 'checkpoint-epoch=0014.ckpt'

module = WhisperModelModule(config)
try:
    state_dict = torch.load(config.checkpoint_path, map_location=torch.device(device))
    state_dict = state_dict["state_dict"]
    module.load_state_dict(state_dict)
    # print(f"load checkpoint successfully from {config.checkpoint_path}")
except Exception as e:
    print(e)
    print(f"load checkpoint failt using origin weigth of {config.model_name} model")
base_model = module.model
base_model.to(device)


class language(str, Enum):
    en = 'English'
    vi = 'Vietnamese'

class version(str, Enum):
    small = 'small'
    base = 'base'



@app.post("/Whisper/")
def speech_to_text(file: UploadFile, language: language, version: version):
    audio_path = file.filename

    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    options = whisper.DecodingOptions(
    language=language, without_timestamps=True, fp16=torch.cuda.is_available()
)
    if version == 'small':
        result = whisper.decode(model, mel, options)
    elif version == 'base':
        result = base_model.decode(mel, options)
    
    return {"transcribe": result.text}

