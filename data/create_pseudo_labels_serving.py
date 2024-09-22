import os
from typing import List
import torch
import argparse

import glob
from tqdm import tqdm
import json
# from vllm import LLM, SamplingParams
from openai import OpenAI


# Modify OpenAI's API key and API base to use vLLM's API server.

openai_api_key = "EMPTY"

openai_api_base = "http://localhost:8000/v1"
def custom_chat_template_llama(messages):
    template = ""
    for msg in messages:
        if msg["role"] == "user":
            template += f"<|start_header_id|>user<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
        elif msg["role"] == "assistant":
            template += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
    # Add the final assistant prompt
    template += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return template

def custom_chat_template_qwen(messages):
    template = ""
    for msg in messages:
        if msg["role"] == "user":
            template += f"<|im_start|>user\n\n{msg['content']}<|im_end|>"
        elif msg["role"] == "assistant":
            template += f"<|im_start|>assistant\n\n{msg['content']}<|im_end|>"
    # Add the final assistant prompt
    template += "<|im_start|>assistant\n\n"
    return template

def handle_conversations(conversations: List[str],model,eos_token,conversation_fn, client: OpenAI):
    
    current_length = 0
    chats = []
    for i in range(0, len(conversations), 2):
        chats.append({
            "role": "user",
            "content": conversations[i]
        })
        prompt = conversation_fn(chats)
        completion = client.completions.create(

            model=model,

            prompt=prompt,

            echo=False,

            n=1,

            stream=False,

            logprobs=1,
            temperature=0.0,
            top_p = 1,
            max_tokens=2048,
            stop=eos_token)
        generated_text = completion.choices[0].text
        chats.append({
            "role": "assistant",
            "content": generated_text
        })
        current_length = completion.usage.total_tokens
        if current_length >= 2048:
            break
    return chats



def process_file(file,  output_dir, from_pct,to_pct):
    try:
        
        client = OpenAI(

            # defaults to os.environ.get("OPENAI_API_KEY")

            api_key=openai_api_key,

            base_url=openai_api_base,

        )
        models = client.models.list()

        model = models.data[0].id
        conversation_fn = custom_chat_template_llama if 'llama' in model.lower() else custom_chat_template_qwen
        eos_token = '<|eot_id|>' if 'llama' in model.lower() else '<|im_end|>'
        with open(file, 'r') as f:
            lines = f.readlines()
        from_idx = int(from_pct * len(lines))
        to_idx = int(to_pct * len(lines))
        lines = lines[from_idx:to_idx]
        progress_bar = tqdm(lines, desc='processing ' + file)
        output_file_name = os.path.basename(file)
        output_file_name = os.path.join(output_dir, output_file_name)
        with open(output_file_name, 'w') as f:
            for line in progress_bar:
                line = line.strip()
                if len(line) == 0:
                    continue
                conversations = json.loads(line)['data']
                new_conversations = handle_conversations(conversations,model,eos_token, conversation_fn,client)
                f.write(json.dumps({'data': new_conversations}) + '\n')
    except Exception as e:
        print(f"Error processing file {file} on device {i}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/rwkv/data/ultrachat')
    parser.add_argument('--output_dir', type=str, default='/data/rwkv/data/ultrachat_llama3_1_pseudo_labels/')
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--from_percent',type=float,default=0.0)
    parser.add_argument('--to_percent',type=float,default=1.0)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    num_devices = args.num_devices
    from_pct = args.from_percent
    to_pct = args.to_percent
    os.makedirs(output_dir, exist_ok=True)

    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # device = args.device
    # model = AutoModelForCausalLM.from_pretrained(model_id).to(dtype=torch.float16, device=device)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     model.config.pad_token_id = model.config.eos_token_id
    files = glob.glob(os.path.join(data_dir, '*.jsonl'))
    print('processing ', files)
    # 创建进程池
    import multiprocessing
    pool = multiprocessing.Pool(processes=num_devices)
    # 为每个文件分配一个进程和一个设备
    for i, file in enumerate(files):
        pool.apply_async(process_file, args=(file, output_dir,from_pct,to_pct))

    pool.close()
    pool.join()

if __name__ == '__main__':
    main()