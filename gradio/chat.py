import os
import sys
import psutil

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(f'add {project_root} to sys.path')

import gradio as gr
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from rwkv_llama.utilities import HybridCache
from rwkv_llama.hybrid_model_run import create_rwkv_args, HybridModel



# 全局变量
model = None
tokenizer = None
def create_new_session():
    return {
        "conversation": [],
        "cache": HybridCache()
    }

def load_model(config_file, ckpt_file):
    global model, tokenizer, cache
    
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model_id = config['Llama']['model_id']
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    transformer_model = AutoModelForCausalLM.from_pretrained(model_id)
    
    args = create_rwkv_args(transformer_model.config, config)
    model = HybridModel(transformer_model, args)
    model.load_ckpt(ckpt_file)
    model = model.to(dtype=torch.bfloat16, device="cuda:0")
    model.eval()
    
    print(model)    
    return "模型加载成功!"

def chat(message, history, session):
    global model, tokenizer
    print(message)
    
    if session is None:
        print("create new session")
        session = create_new_session()
    
    session["conversation"].append({
        'role': 'user',
        'content': message
    })
    print(session["conversation"])
    current_input_text = tokenizer.apply_chat_template(session["conversation"], tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(current_input_text, return_tensors="pt").to("cuda:0")
    input_length = input_ids.input_ids.shape[1]
    with torch.no_grad():
        output = model.model.generate(
            input_ids=input_ids['input_ids'],
            attention_mask=input_ids['attention_mask'],
            max_new_tokens=2048,
            num_return_sequences=1,
            past_key_values=session["cache"],
            use_cache=True,
            early_stopping=True,
            do_sample=True,
        )
    
    generated_text = tokenizer.decode(output[0,input_length:], skip_special_tokens=True)            
    
    session["conversation"].append({
        'role': 'assistant',
        'content': generated_text
    })
    
    print(generated_text)
    return history + [[message, generated_text]], session

import gradio as gr

config_file = "/home/yueyulin/github/RWKVinLLAMA/configs/step_wise/test_hybrid_5_layer_qwenmlp_local.yaml"
ckpt_file = "/home/yueyulin/model/qwen/layer5.pth"
import argparse
parser = argparse.ArgumentParser()  
parser.add_argument('--config_file', type=str, default=config_file)
parser.add_argument('--ckpt_file', type=str, default=ckpt_file)
args = parser.parse_args()
load_model(args.config_file, args.ckpt_file)
def clear_cache(session):
    if session is not None and "cache" in session:
        del session["cache"]
        torch.cuda.empty_cache()
        session["cache"] = HybridCache()
    return session
def clear_conversation(session):
    if session is not None:
        session["conversation"] = []
        del session["cache"]
        torch.cuda.empty_cache()
        session["cache"] = HybridCache()
    return [], session
def get_memory_usage():
    memory = psutil.virtual_memory()
    cuda_memory = get_cuda_memory_usage()
    return f"总内存: {memory.total / (1024**3):.2f} GB\n已使用: {memory.used / (1024**3):.2f} GB\n可用: {memory.available / (1024**3):.2f} GB\n使用率: {memory.percent}%\n{cuda_memory}"
def get_cuda_memory_usage():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)
        memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)
        memory_free = torch.cuda.get_device_properties(device).total_memory / (1024**2) - memory_reserved
        return f"CUDA显存:\n已分配: {memory_allocated:.5f} MB\n已预留: {memory_reserved:.5f} MB\n可用: {memory_free:.5f} MB"
    else:
        return "CUDA不可用"

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown(f"当前配置文件: {args.config_file}\n当前检查点文件: {args.ckpt_file}")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear_chat_btn = gr.Button("清除对话")
    clear_cache_btn = gr.Button("清除缓存")
    session = gr.State()
    
    memory_info = gr.Textbox(label="系统内存和CUDA显存使用情况")
    update_memory_btn = gr.Button("更新内存信息")
    
    msg.submit(chat, inputs=[msg, chatbot, session], outputs=[chatbot, session])
    clear_chat_btn.click(clear_conversation, inputs=[session], outputs=[chatbot, session])
    clear_cache_btn.click(clear_cache, inputs=[session], outputs=[session])
    
    update_memory_btn.click(get_memory_usage, outputs=memory_info)

demo.launch(server_name="0.0.0.0", server_port=7860)
