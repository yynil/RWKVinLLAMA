import os
import sys
import psutil

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(f"add {project_root} to sys.path")

import gradio as gr
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from rwkv_llama.utilities import HybridCache
from rwkv_llama.hybrid_model_run_rwkv7 import create_rwkv_args, HybridModel
from transformers.modeling_utils import no_init_weights


# 全局变量
model = None
tokenizer = None
is_hybrid = False


def create_new_session():
    return {"conversation": [], "cache": HybridCache()}


def load_model(config_file, ckpt_file, num_gpus, off_load_emb_head):
    global model, tokenizer, cache

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_id = config["Llama"]["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16
    if is_hybrid:
        transformer_config = AutoConfig.from_pretrained(model_id)
        args = create_rwkv_args(transformer_config, config)
        print(f"args is {args}")
        with no_init_weights():
            model = HybridModel(args, transformer_config)
        if ckpt_file is None:
            #try to load from model_id
            model.load_checkpoint(model_id)
        else:
            model.load_checkpoint(ckpt_file)
        from accelerate import dispatch_model, infer_auto_device_map

        model = model.to(dtype=dtype)
        num_layers = model.model.config.num_hidden_layers
        device_map = {}
        average_layers = num_layers // num_gpus
        for i in range(num_layers):
            device_map[f"model.layers.{i}"] = i // average_layers
        if off_load_emb_head:
            device_map["model.embed_tokens"] = "cpu"
            device_map["model.norm"] = "cpu"
            device_map["model.rotary_emb"] = "cpu"
            device_map["lm_head"] = "cpu"
        else:
            device_map["model.embed_tokens"] = 0
            device_map["model.norm"] = num_gpus - 1
            device_map["model.rotary_emb"] = num_gpus - 1
            device_map["lm_head"] = num_gpus - 1
        # device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1, 'model.layers.32': 1, 'model.layers.33': 2, 'model.layers.34': 2, 'model.layers.35': 2, 'model.layers.36': 2, 'model.layers.37': 2, 'model.layers.38': 2, 'model.layers.39': 2, 'model.layers.40': 2, 'model.layers.41': 2, 'model.layers.42': 2, 'model.layers.43': 2, 'model.layers.44': 2, 'model.layers.45': 2, 'model.layers.46': 2, 'model.layers.47': 2, 'model.layers.48': 2, 'model.layers.49': 2, 'model.layers.50': 3, 'model.layers.51': 3, 'model.layers.52': 3, 'model.layers.53': 3, 'model.layers.54': 3, 'model.layers.55': 3, 'model.layers.56': 3, 'model.layers.57': 3, 'model.layers.58': 3, 'model.layers.59': 3, 'model.layers.60': 3, 'model.layers.61': 3, 'model.layers.62': 3, 'model.layers.63': 3, 'model.norm': 3, 'model.rotary_emb': 3, 'lm_head': 3}
        model.model = dispatch_model(
            model.model, device_map=device_map, offload_buffers=True
        )
        print("Model is HYBRID!")
        print(model)
    else:
        transformer_model = AutoModelForCausalLM.from_pretrained(model_id)
        model = transformer_model
        model = model.to(dtype=torch.bfloat16, device="cuda:0")
    model.eval()

    print(model)
    return "模型加载成功!"


def chat(message, history, session):
    global model, tokenizer, is_hybrid
    print(message)

    if session is None:
        print("create new session")
        session = create_new_session()

    session["conversation"].append({"role": "user", "content": message})
    print(session["conversation"])
    current_input_text = tokenizer.apply_chat_template(
        session["conversation"], tokenize=False, add_generation_prompt=True
    )
    # print(current_input_text)
    index_of_im_start = current_input_text.find("<|im_start|>user")
    if index_of_im_start != -1:
        current_input_text = current_input_text[index_of_im_start:]
    print(current_input_text)
    input_ids = tokenizer(current_input_text, return_tensors="pt").to("cuda:0")
    input_length = input_ids.input_ids.shape[1]
    from transformers import GenerationConfig

    gen_config = GenerationConfig(
        max_new_tokens=1024,
        stop_strings=["<|im_end|>"],
        do_sample=True,
        use_cache=True,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        min_p=0.05,
        repetition_penalty=1.1,
        no_repeat_ngram_size=4,
    )
    print(f'cache is {session["cache"]}')
    with torch.no_grad():
        if is_hybrid:
            print("use hybrid model to generate")
            model_to_use = model.model
        else:
            model_to_use = model
        print(f'{input_ids['input_ids'].shape}')
        output = model_to_use.generate(
            input_ids=input_ids["input_ids"],
            attention_mask=input_ids["attention_mask"],
            past_key_values=session["cache"],
            generation_config=gen_config,
            tokenizer=tokenizer,
            output_attentions=False,
            use_cache=True,
        )
        
    generated_tokens_length = output.shape[1] - input_length
    print(f'generated {generated_tokens_length}')
    print(f'generated ids {output[0, input_length:]}')

    generated_text = tokenizer.decode(
        output[0, input_length:], skip_special_tokens=True
    )

    session["conversation"].append({"role": "assistant", "content": generated_text})

    print(generated_text)
    return history + [[message, generated_text]], session


import gradio as gr

config_file = "/home/yueyulin/github/RWKVinLLAMA/configs/step_wise/test_hybrid_5_layer_qwenmlp_local.yaml"
ckpt_file = "/home/yueyulin/model/qwen/layer5.pth"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default=config_file)
parser.add_argument("--ckpt_file", type=str, default=None)
parser.add_argument("--is_hybrid", action="store_true", default=False)
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--off_load_emb_head", action="store_true", default=False)
args = parser.parse_args()
print(args)
is_hybrid = args.is_hybrid
load_model(args.config_file, args.ckpt_file, args.num_gpus, args.off_load_emb_head)


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
        memory_free = (
            torch.cuda.get_device_properties(device).total_memory / (1024**2)
            - memory_reserved
        )
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
    clear_chat_btn.click(
        clear_conversation, inputs=[session], outputs=[chatbot, session]
    )
    clear_cache_btn.click(clear_cache, inputs=[session], outputs=[session])

    update_memory_btn.click(get_memory_usage, outputs=memory_info)

demo.launch(server_name="0.0.0.0", server_port=7860)
