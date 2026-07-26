import torch
from dataloader import load_resized, data
from tqdm import tqdm

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
    no_cls = fwd_pass.last_hidden_state[:, 1:].mean(dim=1)

    
    z1 = torch.nn.functional.normalize(cls, p=2, dim=1)
    z2 = torch.nn.functional.normalize(no_cls, p=2, dim=1)                                                                                                                                                                                                                                                 
    return z1.detach().cpu().squeeze(0), z2.detach().cpu().squeeze(0)


##########################################################################################


embeddings_x, embeddings_y = {}, {}  

for family, paths in tqdm(data.items(), desc="families"):    #list(data.items())[:x] for small sample testing
    vecs_x, vecs_y = [], []
    for p in paths:
        x,y = embed_image(p, model, processor)
        vecs_x.append(x)
        vecs_y.append(y)
    embeddings_x[family] = torch.stack(vecs_x)  # [n_images, 2048]
    embeddings_y[family] = torch.stack(vecs_y)

torch.save(embeddings_x, "dino_cls_malimg_emb.pt")   # embeddings = {family: [n, 2048] matrix}
torch.save(embeddings_y, "dino_noncls_malimg_emb.pt")

for family, matrix in embeddings_x.items():
    print(family, matrix.shape)

for family, matrix in embeddings_y.items():
    print(family, matrix.shape)
