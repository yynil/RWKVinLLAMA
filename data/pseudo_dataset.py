import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, training_data, training_label, pad_token_id):
        self.training_data = training_data
        self.training_label = training_label
        self.pad_token_id = pad_token_id
        assert self.training_data.shape == self.training_label.shape
        
    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        input_ids = self.training_data[idx]
        return {"input_ids": input_ids, "labels": self.training_label[idx], "attention_mask": input_ids.ne(self.pad_token_id)}
    
if __name__ == '__main__':
    input_ids_file = '/home/yueyulin/data/preprocessed_data/ultrachat/input_ids.pt'
    labels_ids_file = '/home/yueyulin/data/preprocessed_data/ultrachat/labels.pt'
    pad_token_id = 128009
    
    ds = TextDataset(torch.load(input_ids_file), torch.load(labels_ids_file), pad_token_id)
    
    print(ds[0])
    input_ids = ds[0]['input_ids']
    labels = ds[0]['labels']
    labels[labels == -100] = pad_token_id
    model_id = '/home/yueyulin/model/llama-3.1-8B-Instruct/'
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(tokenizer.decode(input_ids))
    print('=================================')
    print(tokenizer.decode(labels))
    print(input_ids.tolist())
    input_str = tokenizer.decode(input_ids)
    user_tag = "<|start_header_id|>user<|end_header_id|>\n\n"
    assistant_tag = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    tags = [user_tag, assistant_tag]
    offset = 0
    conversations = []
    while offset < len(input_str):
        start = input_str.find(user_tag, offset)
        if start == -1:
            break
        end = input_str.find(assistant_tag, start)
        if end == -1:
            break
        conversation = input_str[start+len(user_tag):end]
        offset = end + len(assistant_tag)
        conversations.append(conversation)
        next_start = input_str.find(user_tag, offset)
        if next_start == -1:
            #no more user tag
            response = input_str[offset:]
            conversations.append(response)
            break
        else:
            response = input_str[offset:next_start]
            conversations.append(response)
            offset = next_start

    role = ['🐱','🤖']
    device = 'cuda:0'
    model = AutoModelForCausalLM.from_pretrained(model_id).to(dtype=torch.float16,device=device)
    all_input_str = ""
    prompts = []
    responses = []
    for i in range(len(conversations)):
        if i % 2 == 0:
            all_input_str += tags[0] + conversations[i] + tags[1]
            with torch.no_grad():
                input_ids = tokenizer(all_input_str, return_tensors="pt").to(device)
                output = model.generate(
                    input_ids = input_ids['input_ids'],
                    attention_mask = input_ids['attention_mask'],
                    max_length=4096,
                    num_return_sequences=1,
                    use_cache=True,stop_strings=["<|eot_id|>"], tokenizer=tokenizer,
                    top_k = 1
                )
                output_ids = output[0][input_ids['input_ids'].shape[1]:]
                generated_text = tokenizer.decode(output_ids)
                all_input_str += generated_text+'\n\n'
                prompts.append(conversations[i])
                responses.append(generated_text)
    print('-----------')
    import json
    print(json.dumps(prompts, indent=4))
    print('-----------')
    print(json.dumps(responses, indent=4))
    