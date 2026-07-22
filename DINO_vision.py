import torch
from dataloader import load_resized, data
from tqdm import tqdm

import requests
#from torchao.quantization import Int4WeightOnlyConfig
from transformers import AutoImageProcessor, AutoModel



print(torch.cuda.is_available())

model = AutoModel.from_pretrained(
    "facebook/dinov2-base",
    device_map="auto",
    attn_implementation="sdpa"
)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')


def embed_image(path, model, processor):

    x = load_resized(path)

    inputs = processor(x, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        fwd_pass = model(**inputs)

    cls = fwd_pass.last_hidden_state[:, 0]  # grabbing CLS

    
    z = torch.nn.functional.normalize(cls, p=2, dim=1)                                                                                                                                                                                                                                                          
    return z.detach().cpu().squeeze(0)


##########################################################################################


embeddings = {}  

for family, paths in tqdm(data.items(), desc="families"):    #list(data.items())[:x] for small sample testing
    vecs = []
    for p in paths:
        vecs.append(embed_image(p, model, processor))
    embeddings[family] = torch.stack(vecs)  # [n_images, 2048]

torch.save(embeddings, "dino_malimg_embeddings.pt")   # embeddings = {family: [n, 2048] matrix}

for family, matrix in embeddings.items():
    print(family, matrix.shape)