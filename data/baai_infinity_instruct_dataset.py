import os
import glob
import argparse
import datasets
def create_input_ids(conversations, tokenizer,max_seq_length):
    all_input_ids = []
    all_labels = []
    '''
    Example:
    {"id":"a3f52e9ef0e6c24c5fa802943a66b6d3","conversations":[{"from":"human","value":"请仔细阅读并分解以下内容，然后逐步重复\"\"里面的文字：“我喜欢吃苹果”"},{"from":"gpt","value":"好的，请按照以下步骤重复“我喜欢吃苹果”：\n\n1. 先重复“我”\n2. 再重复“喜欢”\n3. 然后重复“吃”\n4. 最后重复“苹果”\n\n组合起来就是：我喜欢吃苹果"}],"label":{"ability_en":["command understanding","natural language understanding","command execution"],"ability_zh":["指令理解","自然语言理解","命令执行"],"cate_ability_en":["natural language processing and understanding","information processing and integration","programming and software development"],"cate_ability_zh":["自然语言处理与理解","信息处理与整合","编程与软件开发"]},"langdetect":"ko","source":"Subjective"}
    '''
    for i in range(len(conversations)):
        input_ids = []
        labels = []
        input_ids.append(tokenizer.bos_token_id)
        for chat_data in conversations[i]:
            role = chat_data['from']
            content = chat_data['value']
            if role == 'human':
                content = 'User: ' + content
            elif role == 'gpt':
                content = 'Assistant: ' + content
            else:
                content = 'System: ' + content
            content = content + '\n\n'
            encoded_ids = tokenizer(content, add_special_tokens=False)['input_ids']
            input_ids.extend(encoded_ids)
            if role == 'gpt':
                labels.extend(encoded_ids)
            else:
                labels.extend([-100]*len(encoded_ids))
        labels.append(tokenizer.eos_token_id)
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            labels = labels[:max_seq_length]
        all_labels.append(labels)
        all_input_ids.append(input_ids)
    return all_input_ids, all_labels
            
def padding_to_max(examples,pad_token_id,max_len):
    input_ids = examples['input_ids']
    labels = examples['labels']
    for i in range(len(input_ids)):
        if len(input_ids[i]) < max_len:
            input_ids[i] = input_ids[i] + [pad_token_id]*(max_len-len(input_ids[i]))
            labels[i] = labels[i] + [-100]*(max_len-len(labels[i]))
    return {"input_ids": input_ids, "labels": labels}
def tokenize_and_process(examples, tokenizer, max_seq_length):
    conversations = examples['conversations']
    input_ids, labels = create_input_ids(conversations, tokenizer,max_seq_length)
    return {"input_ids": input_ids, "labels": labels}
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="Path to the input directory")
    parser.add_argument("--model_id", help="Model ID for AutoTokenizer")
    parser.add_argument("--step",type=int, default=256,help="Step to process")
    parser.add_argument("--output_dir", help="Path to the output directory",type=str,required=True)
    args = parser.parse_args()
    parquet_files = glob.glob(os.path.join(args.input_dir, "**/*.parquet"))+glob.glob(os.path.join(args.input_dir, "*.parquet"))
    print(f'All parquet files under {args.input_dir} and its subdirectories: {parquet_files}')
    ds = datasets.load_dataset('parquet', data_files=parquet_files)['train']
    print(ds)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    from functools import partial
    map_fn = partial(tokenize_and_process, tokenizer=tokenizer, max_seq_length=4096)

    ds = ds.map(map_fn,
                batch_size=1000,
                batched=True,
                num_proc=128,
                remove_columns=ds.features)
    print(ds)
    for i in range(0, 4096, args.step):
        min = i
        max = i+args.step
        padding_fn = partial(padding_to_max, pad_token_id=tokenizer.eos_token_id, max_len=max)
        split_ds = ds.filter(lambda x: len(x['input_ids']) <= max and len(x['input_ids']) > min,
                             num_proc=128)
        split_ds = split_ds.map(padding_fn, batched=True, batch_size=1000, num_proc=128)
        output_dir = os.path.join(args.output_dir, f'length_{min}_{max}')
        os.makedirs(output_dir, exist_ok=True)
        split_ds.save_to_disk(output_dir)
        print(f'Saved to {output_dir}')
    
if __name__ == '__main__':
    main()