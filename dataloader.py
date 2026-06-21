import pathlib
from pathlib import Path
import os

dir = pathlib.Path(r"C:\Users\soul4\OneDrive\Desktop\qwen_replicate\malimg_paper_dataset_imgs")

data = {}
for family_dir in dir.iterdir():
    if not family_dir.is_dir():
        continue
    images = sorted(family_dir.glob("*.png"))
    if images:
        data[family_dir.name] = images


#print(data.keys())
import sys
from PIL import Image, ImageOps

TARGET_SIZE = (384, 384)

def load_resized(path):
    img = Image.open(path).convert("RGB")   # grayscale -> 3-channel
    return img.resize(TARGET_SIZE)  

print("Dataloader OK!")