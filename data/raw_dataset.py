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
    
    directory = '/home/yueyulin/data/sftdatasetv3/'
    dataset_type = detect_format(directory)
    print(f"Detected dataset type: {dataset_type}")
    dataset = load_functions[dataset_type](directory)
    print(dataset)
    print(check_feature(dataset))
    
    directories = ['/home/yueyulin/data/finemath/finemath-4plus/', '/home/yueyulin/data/sftdatasetv3/']
    