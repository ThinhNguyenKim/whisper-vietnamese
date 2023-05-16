import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import os
from datetime import datetime
from typing import Optional
import time
from loguru import logger
import shutil

import whisper
import torch
from asr.config import Config
from asr.model import WhisperModelModule

from asr.utils import hf_to_whisper_states

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
async def speech_to_text(
    file: UploadFile,
    language: language,
    version: version,
    beam_size: Optional[int]=None
    ):
    file_location = 'record.wav'
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    audio = whisper.load_audio(file_location)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    options = whisper.DecodingOptions(
    language=language, without_timestamps=True,fp16=torch.cuda.is_available(), beam_size=beam_size
)
    if version == 'small':
        
        start_time = time.time()
        
        result = whisper.decode(model, mel, options)
        
        end_time = time.time()
        total_time = str(end_time - start_time)
        
        logger.success(total_time)
    
    elif version == 'base':
        result = base_model.decode(mel, options)
    
    return {"transcribe": result.text}

if __name__ == '__main__':
    uvicorn.run("app:app", reload=True, host='0.0.0.0', port=8080)

