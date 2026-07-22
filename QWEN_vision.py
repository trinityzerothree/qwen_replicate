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
    inputs = processor(text=[text], images=[x], padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        fwd_pass = model(**inputs, output_hidden_states=True)

    layer_combined = torch.stack(fwd_pass.hidden_states[-4:]).mean(dim=0)  # [4,1,seq_len, 2048], mfirst dim gets collapsed
    pooled = layer_combined.mean(dim=1)
    
    
    z = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return z.detach().cpu().squeeze(0)


##########################################################################################


embeddings = {}  

for family, paths in tqdm(data.items(), desc="families"):    #list(data.items())[:x] for small sample testing
    vecs = []
    for p in paths:
        vecs.append(embed_image(p, model, processor))
    embeddings[family] = torch.stack(vecs)  # [n_images, 2048]

torch.save(embeddings, "malimg_embeddings.pt")   # embeddings = {family: [n, 2048] matrix}

for family, matrix in embeddings.items():
    print(family, matrix.shape)