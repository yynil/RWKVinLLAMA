model_id = '/home/yueyulin/model/llama-3.1-8B-Instruct/'
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
B,T=2,512

model = AutoModelForCausalLM.from_pretrained(model_id).to(dtype=torch.bfloat16,device='cuda:0')
input_ids = torch.randint(0, 128255, (B, T), dtype=torch.long).to('cuda:0')
print(input_ids)
output = model(input_ids,output_hidden_states=True)
print(output.logits.shape)
print(output.hidden_states[-1].shape)
del input_ids
del output

torch.cuda.empty_cache()
input_ids = torch.randint(0, 128255, (B, T), dtype=torch.long).to('cuda:0')
output = model(input_ids,output_hidden_states=True)
print(output.logits.shape)
print(output.hidden_states[-1].shape)
