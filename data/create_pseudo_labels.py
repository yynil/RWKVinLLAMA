import os
from typing import List
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList,DynamicCache
import glob
from tqdm import tqdm
import json
class EOSTokenStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1].item() == self.eos_token_id
def handle_conversations(conversations: List[str], model, tokenizer, device):
    new_conversations = []
    user_tag = "<|start_header_id|>user<|end_header_id|>\n\n"
    assistant_tag = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    past_key_values = DynamicCache()
    # Create a stopping criteria object
    stopping_criteria = StoppingCriteriaList([EOSTokenStoppingCriteria(tokenizer.eos_token_id)])
    new_input = ""
    whole_id_counts = 0
    for i in range(len(conversations)):
        if i % 2 == 0:
            # Only tokenize the new user input
            new_input += user_tag + conversations[i]+'<|eot_id|>' + assistant_tag
            inputs = tokenizer(new_input, return_tensors="pt").to(device)

            whole_id_counts = inputs['input_ids'].shape[1]
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=2048,  # Adjust as needed
                    num_return_sequences=1,
                    use_cache=True,
                    past_key_values=past_key_values,
                    return_dict_in_generate=True,
                    output_scores=False,
                    top_k=1,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria 
                )
            generated_ids = output.sequences[0][inputs['input_ids'].shape[1]:]
            generated_text = tokenizer.decode(generated_ids)
            past_key_values = output.past_key_values  # Update past_key_values

            new_conversations.append(
                {"role": "user", "content": conversations[i]+'<|eot_id|>'})
            new_conversations.append(
                {"role": "assistant", "content": generated_text}
            )
            if output.sequences[0].shape[0] > 2048:
                break
            new_input += generated_text

    return new_conversations

def process_file(file, model_id, output_dir, device,from_pct,to_pct):
    try:
        print(f'start process for {file} / {model_id} {device},{output_dir}, {from_pct}, {to_pct}')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(dtype=torch.float16, device=device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        with open(file, 'r') as f:
            lines = f.readlines()
        from_idx = int(from_pct * len(lines))
        to_idx = int(to_pct * len(lines))
        lines = lines[from_idx:to_idx]
        progress_bar = tqdm(lines, desc='processing ' + file)
        output_file_name = os.path.basename(file)
        output_file_name = f'{output_file_name[:-6]}_pseudo_labels_{from_pct}_{to_pct}.jsonl'
        output_file_name = os.path.join(output_dir, output_file_name)
        with open(output_file_name, 'w') as f:
            for line in progress_bar:
                line = line.strip()
                if len(line) == 0:
                    continue
                conversations = json.loads(line)['data']
                new_conversations = handle_conversations(conversations, model, tokenizer, device)
                if len(new_conversations) > 0:
                    f.write(json.dumps({'data': new_conversations}) + '\n')
    except Exception as ex:
        print(ex)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='/data/rwkv/models/meta-llama/Meta-Llama-3.1-8B-Instruct/')
    parser.add_argument('--data_dir', type=str, default='/data/rwkv/data/ultrachat')
    parser.add_argument('--output_dir', type=str, default='/data/rwkv/data/ultrachat_llama3_1_pseudo_labels/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--from_percent',type=float,default=0.0)
    parser.add_argument('--to_percent',type=float,default=1.0)
    args = parser.parse_args()

    model_id = args.model_id
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
        device = f'cuda:{i % num_devices}'
        pool.apply_async(process_file, args=(file, model_id, output_dir, device,from_pct,to_pct))

    pool.close()
    pool.join()

if __name__ == '__main__':
    main()