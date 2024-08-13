if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, default='/media/yueyulin/data_4t/models/tinyLlama-1.1B-Chat-V1.0/')
    parser.add_argument('--output_dir', type=str, default='/media/yueyulin/data_4t/data/ultrachat_200k_ds/')
    args = parser.parse_args()
    import os
    os.environ['HF_ENDPOINT']='https://hf-mirror.com'
    import datasets
    from datasets import load_dataset
    dataset_name = 'HuggingFaceH4/ultrachat_200k'
    dataset = load_dataset(dataset_name)
    print(dataset)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(tokenizer)
    import json
    print(json.dumps(dataset['train_sft'][17]))
    print(json.dumps(dataset['train_gen'][17]))
    print(tokenizer.encode('assistant',add_special_tokens=False))
    print(tokenizer.encode('Yes',add_special_tokens=False))
    print(tokenizer.encode('yes',add_special_tokens=False))
    max_len = 2048
    
    def tokenize_and_cut(examples):
        inputs_ids = []
        labels_ids = []
        for messages in examples['messages']:
            input_ids = [tokenizer.bos_token_id]
            labels = [-100]
            for msg_obj in messages:
                content = msg_obj['content']
                role = msg_obj['role']+' : '
                role_ids = tokenizer.encode(role, add_special_tokens=False)
                ids = tokenizer.encode(content, add_special_tokens=False)
                input_ids.extend(role_ids)
                input_ids.extend(ids)
                if 'assistant' in role:
                    labels.extend([-100]*len(role_ids))
                    labels.extend(ids)
                else:
                    labels.extend([-100]*(len(ids)+len(role_ids)))
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]
            labels = labels[1:]+[tokenizer.eos_token_id]
            input_ids = input_ids+[tokenizer.pad_token_id]*(max_len-len(input_ids))
            labels = labels+[-100]*(max_len-len(labels))
            inputs_ids.append(input_ids)
            labels_ids.append(labels)
        return {'input_ids':inputs_ids,'labels':labels_ids}
    train_sft_ds = dataset['train_sft'].map(tokenize_and_cut, batched=True,remove_columns=['prompt','messages','prompt_id'],num_proc=4)
    os.makedirs(args.output_dir,exist_ok=True)
    train_sft_ds.save_to_disk(args.output_dir)
                