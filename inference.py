# try:
#     import tensorflow  # required in Colab to avoid protobuf compatibility issues
# except ImportError:
#     pass
import warnings
warnings.filterwarnings('ignore')

import whisper
import torch
import argparse

from config import Config
from model import WhisperModelModule
from utils import load_wave, hf_to_whisper_states



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint-epoch=0014.ckpt', help='path of checkpoint, if not set, use origin pretrained model')
    parser.add_argument('--audio_path', type=str, default='thinh_en.m4a', help='the audio file for inference')
    parser.add_argument('--model_name', type=str, default='small', help='model name')
    parser.add_argument('--language', type=str, default='en', help='language')

    args = parser.parse_args()
    config = Config()
    if ".bin" in args.checkpoint_path:
        hf_state_dict = torch.load(args.checkpoint_path, map_location=torch.device(device))

        for key in list(hf_state_dict.keys())[:]:
            new_key = hf_to_whisper_states(key)
            hf_state_dict[new_key] = hf_state_dict.pop(key)
        
        model = whisper.load_model(args.model_name)
        model.load_state_dict(hf_state_dict)
        model.to(device)
    else:
        config.checkpoint_path = args.checkpoint_path
        config.model_name = args.model_name

        module = WhisperModelModule(config)
        try:
            state_dict = torch.load(config.checkpoint_path)
            state_dict = state_dict["state_dict"]
            module.load_state_dict(state_dict)
            print(f"load checkpoint successfully from {config.checkpoint_path}")
        except Exception as e:
            print(e)
            print(f"load checkpoint failt using origin weigth of {config.model_name} model")
        model = module.model
        model.to(device)

    audio = whisper.load_audio(args.audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # decode the audio
    options = whisper.DecodingOptions(
        language=args.language, without_timestamps=True, fp16=torch.cuda.is_available()
    )

    result = model.decode(mel, options)
    print('Predicted:', result.text)

    