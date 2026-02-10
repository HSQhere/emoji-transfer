# coding: utf-8
import os
from pathlib import Path

base_path = Path(__file__).parent.absolute()
cache_dir = os.path.join(base_path, "model_cache")
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(cache_dir, "huggingface")
os.environ['MODELSCOPE_CACHE'] = os.path.join(cache_dir, "modelscope")
os.environ['TORCH_HOME'] = os.path.join(cache_dir, "torch")
os.makedirs(cache_dir, exist_ok=True)

import torch
from generator import Generator, MODEL_CONFIG
from PIL import Image

modelscope = False
gpu_id = 0
lora_index = 2
checkpoint_index = 1

lora_names = [None] + list(MODEL_CONFIG['sd15']['loras'].keys())
checkpoint_names = list(MODEL_CONFIG['sd15']['checkpoints'].keys())

if __name__ == '__main__':
    generator = Generator(gpu_id=gpu_id, modelscope=modelscope)

    lora = lora_names[lora_index]
    checkpoint = checkpoint_names[checkpoint_index]

    print(f"Preparing Checkpoint: {checkpoint}")
    print(f"Preparing LoRA: {lora}")

    tmp_lora_info = MODEL_CONFIG['sd15']['loras'][lora]
    if modelscope:
        from modelscope import snapshot_download

        checkpoint_path = snapshot_download(MODEL_CONFIG['sd15']['checkpoints'][checkpoint])
        lora_path = os.path.join(snapshot_download(tmp_lora_info[0]), tmp_lora_info[1]) if lora else None
    else:
        checkpoint_path = MODEL_CONFIG['sd15']['checkpoints'][checkpoint]
        if lora:
            from huggingface_hub import hf_hub_download

            lora_path = hf_hub_download(tmp_lora_info[0], filename=tmp_lora_info[1])
        else:
            lora_path = None

    print("Downloading and loading model to GPU memory, please wait...")
    token = generator.load_pipeline(
        'image',
        checkpoint_path,
        vae_path="same as checkpoint",
        lora_path=lora_path,
        stylize='x1',
        version='v5c'
    )

    print("-" * 30)
    print("[SUCCESS] Model has been fully downloaded and successfully loaded to GPU memory!")
    print(f"Current GPU ID: {gpu_id}")
    print("You can safely close this script, or add image_generate code snippets afterwards for generation.")
    print("-" * 30)