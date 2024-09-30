import sys
import os
def setup_env():
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    rwkv_path = os.path.join(parent_dir, 'rwkv')
    sys.path.append(rwkv_path)
    rwkv_llama_path = os.path.join(parent_dir, 'rwkv_llama')
    sys.path.append(rwkv_llama_path)
    print(f'add path: {rwkv_path} to sys.path')
    print(f'add path: {rwkv_llama_path} to sys.path')
    os.environ['CUDA_HOME'] = '/usr/local/cuda-12.1'
    os.environ['RWKV_JIT_ON'] = '0'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ['RWKV_FLOAT_MODE'] = 'bf16'
    os.environ['RWKV_HEAD_SIZE_A'] = '64'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ["RWKV_MY_TESTING"]='x060'
    os.environ['RWKV_CTXLEN'] = '4096'
    os.environ['WKV'] = 'fla'
    os.environ["RWKV_TRAIN_TYPE"] = ''
setup_env()
import argparse
from argparse import Namespace
import yaml
# parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, default='configs/test_hybrid.yaml')
# args = parser.parse_args()
config_file = 'configs/test_hybrid_full_logits_qwenmlp.yaml'
with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print(config)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
device = 'cuda:0'
dtype = torch.bfloat16
transformer_model = AutoModelForCausalLM.from_pretrained(config['Llama']['model_id'],
                                                         torch_dtype=dtype, device_map={'':'cpu'})
print(transformer_model.config)
rwkv_args = Namespace()
rwkv_args.my_pos_emb = 0
rwkv_args.head_size_a = 64
rwkv_args.head_size_divisor = 8
rwkv_args.ctx_len = 4096
rwkv_args.n_layer = transformer_model.config.num_hidden_layers
rwkv_args.n_embd = transformer_model.config.hidden_size
rwkv_args.dim_att = transformer_model.config.hidden_size
rwkv_args.dim_ffn = transformer_model.config.intermediate_size
rwkv_args.pre_ffn = 0
rwkv_args.head_qk = 0
rwkv_args.tiny_att_dim = 0
rwkv_args.tiny_att_layer = -999
rwkv_args.vocab_size = transformer_model.config.vocab_size
rwkv_args.dropout = 0
rwkv_args.is_llama_ffn = True
rwkv_args.layers = config['RWKV']['layers']
rwkv_args.grad_cp = 0
rwkv_args.is_hidden_align = config['teach_mode']['is_hidden_align']
rwkv_args.is_llama_ffn = config.get('is_llama_ffn', False)
rwkv_args.is_rwkv_att_only = config.get('is_rwkv_att_only', False)
from hybrid_model import HybridModel

model = HybridModel(transformer_model,rwkv_args)
print(model)
if rwkv_args.is_rwkv_att_only:
    print(f'only rwkv att is trained')
    for name, param in model.named_parameters():
        if not 'self_attn.' in name:
            param.requires_grad = False
        print(name, param.shape, param.requires_grad)
else:
    for name, param in model.named_parameters():
        if not 'block.' in name or 'ffn' in name:
            param.requires_grad = False
        print(name, param.shape, param.requires_grad)
model = model.to(device=device, dtype=dtype)
labels = torch.randint(0, rwkv_args.vocab_size, (2, 64), device=device, dtype=torch.long)
labels[0,20:] = -100
labels[1,10:] = -100
input_ids = torch.randint(0, rwkv_args.vocab_size, (2, 64), device=device, dtype=torch.long)
result = model(input_ids,output_hidden_states=True,use_cache=False)
print(result.logits)
print(result.logits.shape)
print(result.hidden_states)
print(result.hidden_states[0].shape)
print(len(result.hidden_states))
hidden_states = torch.cat(result.hidden_states[1:], dim=0)
print(hidden_states.shape)
print(labels)
hidden_states = hidden_states.view(2, rwkv_args.n_layer,64, rwkv_args.n_embd)
mask = torch.ne(labels, -100)
print(mask)
# tokenizer = AutoTokenizer.from_pretrained(config['Llama']['model_id'])
# print(tokenizer.eos_token_id)
mask = mask.unsqueeze(1).unsqueeze(3)
print(mask)
hidden_states = hidden_states * mask
print(hidden_states)
layers = [0,2,4,6,8]
selected_hidden_states = hidden_states[:,layers]
print(selected_hidden_states)
print(selected_hidden_states.shape)
print(hidden_states[:,8])
print(selected_hidden_states[:,4])
# teacher_model = AutoModelForCausalLM.from_pretrained(config['Llama']['model_id'], torch_dtype=dtype)
# teacher_model = teacher_model.to('cuda')
# teacher_model.eval()
# import datasets 
# from datasets import load_from_disk
# ds_dir = '/data/rwkv/data/ultrachat_200k_ds_llama/'
# ds = load_from_disk(ds_dir)
# print(ds)
# input_ids = ds[0]['input_ids']
# labels = ds[0]['labels']
# input_ids = torch.tensor(input_ids,device=device,dtype=torch.long).unsqueeze(0)
# labels = torch.tensor(labels,device=device,dtype=torch.long).unsqueeze(0)
# attention_mask = torch.ne(input_ids, tokenizer.eos_token_id)
# print(input_ids)
# print(labels)
# print(attention_mask)
# print(input_ids.shape)
# print(labels.shape)
# print(attention_mask.shape)
# logits = model(input_ids,attention_mask=attention_mask).logits
# print(logits)
# print(logits.shape)

# with torch.no_grad():
#     teacher_logits = teacher_model(input_ids,attention_mask=attention_mask).logits
#     print(teacher_logits)
#     print(teacher_logits.shape)
# data_file = '/media/yueyulin/data_4t/data/ultra_data/ultrachat/input_ids-002.pt'
# input_ids = torch.load(data_file)
# input_ids_batch = input_ids[:1]
# input_ids_batch = input_ids_batch[:, :512]
# attention_mask = torch.ne(input_ids_batch, tokenizer.eos_token_id)
# print(attention_mask)
# input_ids_batch = input_ids_batch.to(device=device, dtype=torch.long)
# attention_mask = attention_mask.to(device=device, dtype=torch.long)
# output = model(input_ids_batch,attention_mask=attention_mask)
# print('done')
# print(output.logits)
# model = model.to('cpu')
# torch.cuda.empty_cache()
# with torch.no_grad():
#     output_teacher = teacher_model(input_ids_batch,attention_mask=attention_mask)
# print(output_teacher.logits)