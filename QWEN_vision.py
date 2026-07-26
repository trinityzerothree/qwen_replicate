from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from dataloader import load_resized, data
from tqdm import tqdm

print(torch.cuda.is_available())


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


def embed_image(path, model, processor):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "",
                },
                {"type": "text", "text": ""},
            ],
        }
    ]

    x = load_resized(path)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[x], return_tensors="pt").to(model.device)

    #print(inputs['attention_mask'])   #check line to see if i'm using padded tokens or not, even after padding=False
    with torch.inference_mode():
        fwd_pass = model(**inputs, output_hidden_states=True)

    layer_combined = torch.stack(fwd_pass.hidden_states[-4:]).mean(dim=0)  # [4,1,seq_len, 2048], mfirst dim gets collapsed
    pooled = layer_combined.mean(dim=1)
    last_pooled = layer_combined[:, -1]
    
    
    z1 = torch.nn.functional.normalize(pooled, p=2, dim=1)
    z2 = torch.nn.functional.normalize(last_pooled, p=2, dim=1)
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

torch.save(embeddings_x, "qwen_pooled_malimg_emb.pt")   # embeddings = {family: [n, 2048] matrix}
torch.save(embeddings_y, "qwen_lastpooled_malimg_emb.pt")

for family, matrix in embeddings_x.items():
    print(family, matrix.shape)

for family, matrix in embeddings_y.items():
    print(family, matrix.shape)