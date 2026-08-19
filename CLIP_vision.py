import torch
from tqdm import tqdm
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers.image_utils import load_image
from dataloader import load_resized, data

print(torch.cuda.is_available())

model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(path, model, processor):
    x = load_resized(path)
    inputs = processor(images=x, return_tensors="pt", padding=True).to(model.device)   #padding matters for batching text

    with torch.inference_mode():
        fwd_pass = model(**inputs)
 
    z = torch.nn.functional.normalize((fwd_pass.image_embeds), p=2, dim=1)
    return z.detach().cpu().squeeze(0)


embeddings = {}
for family, paths in tqdm(data.items(), desc="families"):
    vecs = []
    for p in paths:
        x = embed_image(p, model, processor)
        vecs.append(x)
    embeddings[family] = torch.stack(vecs)

torch.save(embeddings, "clip_malimg_emb.pt")

for family, matrix in embeddings.items():
    print(family, matrix.shape)
