from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from dataloader import load_resized, data

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

model.eval()

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

x = load_resized(data['Adialer.C'][0])

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
#processor(text, image = x, return_tensor="pt")
torch.cuda.is_available()

inputs = processor(text=[text], images=[x], padding=True, return_tensors="pt")

with torch.inference_mode():
    fwd_pass = model(**inputs, output_hidden_states=True)

print(len(fwd_pass.hidden_states))

print(fwd_pass.hidden_states[-1].shape)