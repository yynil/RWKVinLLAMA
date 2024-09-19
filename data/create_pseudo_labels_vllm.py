import os
from typing import List
import torch
import argparse

import glob
from tqdm import tqdm
import json
from vllm import LLM, SamplingParams
def custom_chat_template(messages):
    template = "<|begin_of_text|>"
    for msg in messages:
        if msg["role"] == "user":
            template += f"<|start_header_id|>user<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
        elif msg["role"] == "assistant":
            template += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
    # Add the final assistant prompt
    template += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return template

def handle_conversations(conversations: List[str], llm: LLM):
    sampling_params = SamplingParams(temperature=0,top_k=1,max_tokens=2048,stop_token_ids=[llm.get_tokenizer().eos_token_id],skip_special_tokens=False,ignore_eos=True)
    chats = []
    for i in range(0, len(conversations), 2):
        chats.append({
            "role": "user",
            "content": conversations[i]
        })
        outputs = llm.generate(custom_chat_template(chats), sampling_params,use_tqdm=False)
        generated_text = outputs[0].outputs[0].text
        chats.append({
            "role": "assistant",
            "content": generated_text
        })
        current_length = len(outputs[0].prompt_token_ids)+len(outputs[0].outputs[0].token_ids)
        if current_length > 2048:
            break
    return chats

def process_file(file, model_id, output_dir, i):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
        llm = LLM(model=model_id, 
                  tensor_parallel_size=1,
                  enable_prefix_caching=True,
                  gpu_memory_utilization=0.9, enable_chunked_prefill=True,)
        with open(file, 'r') as f:
            lines = f.readlines()
        progress_bar = tqdm(lines, desc='processing ' + file)
        output_file_name = os.path.basename(file)
        output_file_name = os.path.join(output_dir, output_file_name)
        with open(output_file_name, 'w') as f:
            for line in progress_bar:
                line = line.strip()
                if len(line) == 0:
                    continue
                conversations = json.loads(line)['data']
                new_conversations = handle_conversations(conversations, llm)
                f.write(json.dumps({'data': new_conversations}) + '\n')
    except Exception as e:
        print(f"Error processing file {file} on device {i}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='/data/rwkv/models/meta-llama/Meta-Llama-3.1-8B-Instruct/')
    parser.add_argument('--data_dir', type=str, default='/data/rwkv/data/ultrachat')
    parser.add_argument('--output_dir', type=str, default='/data/rwkv/data/ultrachat_llama3_1_pseudo_labels/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_devices', type=int, default=1)
    args = parser.parse_args()

    model_id = args.model_id
    data_dir = args.data_dir
    output_dir = args.output_dir
    num_devices = args.num_devices
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
        pool.apply_async(process_file, args=(file, model_id, output_dir,i))

    pool.close()
    pool.join()

if __name__ == '__main__':
    main()