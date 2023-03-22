# Whisper Vietnamese Finetuning

## Installation

1. __clone__ and __cd__ to project location

2. Install libraries

```bash
pip install -r requirements.txt
```

3. For training & inference

```shell
python finetune.py  --model_name base \
                    --dataset fluers

python inference.py --checkpoint_path path/to/ckpt \
                    --audio_path path/to/wav
```

Whisper base [checkpoint](https://drive.google.com/file/d/17-NATrbLRQqXTYNiY3hq_4PwxjXLeLmU/view?usp=share_link) (*base, batch_size 1, gradient accumulation steps 10, epoch 14, lr 0.0001*)\.