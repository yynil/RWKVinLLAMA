from typing import Dict
import datasets
import glob
import os

def detect_format(directory) -> str:
    """
    Detect the format of the dataset in the given directory.
    Args:
    directory: The directory containing the dataset files.
    Returns:
    The format of the dataset.['jsonl', 'parquet', 'huggingface','unknown']
    """
    #if dataset_info.json exists, it is a huggingface dataset
    if os.path.exists(os.path.join(directory, 'dataset_info.json')):
        return 'huggingface'
    #if parquet files exist, it is a parquet dataset
    if glob.glob(os.path.join(directory, '*.parquet')):
        return 'parquet'
    #if jsonl files exist, it is a jsonl dataset
    if glob.glob(os.path.join(directory, '*.jsonl')):
        return 'jsonl'
    return 'unknown'
def load_jsonl_dataset(file_path):
    jsonl_files = glob.glob(file_path+"/*.jsonl")
    print(f'load jsonl files: {jsonl_files}')
    dataset = datasets.load_dataset('json', data_files=jsonl_files)['train']
    return dataset

def load_parquet_dataset(file_path):
    parquet_files = glob.glob(file_path+"/*.parquet")
    print(f'load parquet files: {parquet_files}')
    dataset = datasets.load_dataset('parquet', data_files=parquet_files)['train']
    return dataset

def load_huggingface_dataset(file_path):
    dataset = datasets.load_from_disk(file_path)
    return dataset

def check_feature(dataset):
    """
    check the features of dataset
    if the feature contains messages and the value is a list, return "conversation"
    if the feature contains text and the value is a string, return "text"
    """
    for feature, value in dataset[0].items():
        if feature == 'messages' and isinstance(value, list):
            return 'conversation'
        if feature == 'text' and isinstance(value, str):
            return 'text'
    return 'unknown'

load_functions = {
    'jsonl': load_jsonl_dataset,
    'parquet': load_parquet_dataset,
    'huggingface': load_huggingface_dataset
}

def convert_conversation_to_text(example: Dict) -> Dict:
    """
    Convert a single conversation to ChatML format.
    
    Args:
        example: A dictionary containing a 'messages' list of conversations
        
    Returns:
        A dictionary with a single 'text' key containing the ChatML formatted conversation
    """
    result = []
    for message in example['messages']:
        result.extend([
            f"<|im_start|>{message['role']}\n",
            f"{message['content']}\n",
            "<|im_end|>\n"
        ])
    
    return {'text': "".join(result)}

def convert_conversational_ds_to_text(ds: datasets.Dataset) -> datasets.Dataset:
    """
    Convert a conversational dataset to ChatML format.
    
    Args:
        ds: A dataset containing 'messages' lists of conversations
        
    Returns:
        A dataset with a single 'text' key containing the ChatML formatted conversation
    """
    return ds.map(convert_conversation_to_text,num_proc=8,  # 使用8个进程并行处理
        remove_columns=ds.column_names,  # 移除所有原始列
        desc="Converting conversations"  # 显示进度条描述
    )



def load_datasets_from_directories(directories):
    """
    Load datasets from directories.
    Args:
    directories: A list of directories containing the dataset files.
    Returns:
    A list of datasets.
    """
    all_ds = []
    for directory in directories:
        dataset_type = detect_format(directory)
        print(f"Detected dataset type: {dataset_type}")
        if dataset_type == 'unknown':
            print(f"Unknown dataset type for directory: {directory}")
            continue
        ds = load_functions[dataset_type](directory)
        feature_type = check_feature(ds)
        if feature_type == 'conversation':
            ds = convert_conversational_ds_to_text(ds)
        else:
            ds = ds.select_columns(['text'])
        print(f"Loaded dataset from directory: {directory}")
        all_ds.append(ds)
    return all_ds

if __name__ == '__main__':
    directory = '/home/yueyulin/data/finemath/finemath-4plus/'
    dataset_type = detect_format(directory)
    print(f"Detected dataset type: {dataset_type}")
    dataset = load_functions[dataset_type](directory)
    print(dataset)
    #find unique value of language
    print(dataset.unique('language'))
    print(dataset[0]['text'])
    print(check_feature(dataset))
    dataset = dataset.select_columns(['text'])
    print(dataset[0]['text'])
    print(dataset)
    
    directory = '/home/yueyulin/data/Mobius/standard/'
    dataset_type = detect_format(directory)
    print(f"Detected dataset type: {dataset_type}")
    dataset = load_functions[dataset_type](directory)
    print(dataset)
    print(check_feature(dataset))
    dataset = convert_conversational_ds_to_text(dataset)
    print(dataset)
    print(check_feature(dataset))
    print(dataset[0]['text'])
    directories = ['/home/yueyulin/data/finemath/finemath-4plus/', '/home/yueyulin/data/Mobius/standard/']
    
    all_ds = load_datasets_from_directories(directories)
    print(all_ds)
    con_ds = datasets.concatenate_datasets(all_ds)
    print(con_ds)
    print(con_ds[0]['text'])
    model_path = '/home/yueyulin/model/qwen_7b_stage3_4k_splits/'
    from transformers import DataCollatorForLanguageModeling,AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=4096,
            return_special_tokens_mask=True
        )
    tokenized_dataset = con_ds.map(
        tokenize_function,
        batched=True,
        num_proc=16,
        remove_columns=con_ds.column_names,
        desc="Running tokenization"
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False,
                                                    pad_to_multiple_of=4096)
    import torch
    from torch.utils.data import DataLoader
    data_loader = DataLoader(tokenized_dataset, batch_size=1, collate_fn=data_collator)
    for batch in data_loader:
        print(batch)
        break