from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
def load_and_interleave_c4(local_dir, languages=['en', 'zh'],split='train'):
    datasets = []
    for lang in languages:
        dataset = load_dataset(local_dir, lang,split=split)
        dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'text'])
        datasets.append(dataset)
    
    interleaved_dataset = interleave_datasets(datasets)
    return interleaved_dataset
def data_collator(tokenizer, max_seq_length):
    def collate_fn(examples):
        batch = tokenizer([ex["text"] for ex in examples], padding="max_length", truncation=True, max_length=max_seq_length, return_tensors="pt")
        input_ids, labels = batch["input_ids"], batch["input_ids"].clone()
        
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = tokenizer.eos_token_id
        
        padding_mask = input_ids.eq(tokenizer.eos_token_id)
        input_ids.masked_fill_(padding_mask, tokenizer.pad_token_id)
        labels.masked_fill_(padding_mask, -100)
        
        return {"input_ids": input_ids, "labels": labels}
    
    return collate_fn
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_c4_dir', type=str, required=True, help='local c4 directory')
    parser.add_argument('--languages', type=str, nargs='+', default=['en', 'zh'], help='languages to interleave')
    parser.add_argument('--model_id', type=str, required=True, help='model id')
    args = parser.parse_args()

    local_c4_dir = args.local_c4_dir
    languages = args.languages
    combined_dataset = load_and_interleave_c4(local_c4_dir, languages,split='validation')
    print(combined_dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    collate_fn = data_collator(tokenizer, max_seq_length=2048)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(combined_dataset, collate_fn=collate_fn, batch_size=16, shuffle=True)
    for batch in dataloader:
        print(batch)
        print(batch['input_ids'].shape)
        print(batch['labels'].shape)
        break