from datasets import load_dataset,interleave_datasets
import glob
def load_parquet_dataset(path,split='train', val_size=0.01):
    files = glob.glob(path + '*.parquet') + glob.glob(path + '**/*.parquet')
    print(f'加载 parquet 数据集，文件：{files}')
    dataset = load_dataset('parquet', data_files=files)
    print(f'数据集：{dataset}')
    
    # 分割数据集
    if split == 'train':
        dataset = dataset['train'].train_test_split(test_size=val_size)
        train_dataset = post_process_dataset(dataset['train'], 'train')
        val_dataset = post_process_dataset(dataset['test'], 'validation')
        return train_dataset, val_dataset
    else:
        return post_process_dataset(dataset[split], split)

def load_json_gz_dataset(path, split='train', val_size=0.01):
    files = glob.glob(path + '*.jsonl.gz') + glob.glob(path + '**/*.jsonl.gz')
    print(f'加载 jsonl.gz 数据集，文件：{files}')
    dataset = load_dataset('json', data_files=files)
    print(f'数据集：{dataset}')
    
    # 分割数据集
    if split == 'train':
        dataset = dataset['train'].train_test_split(test_size=val_size)
        print(f'分割后的数据集：{dataset}')
        train_dataset = post_process_dataset(dataset['train'], 'train')
        val_dataset = post_process_dataset(dataset['test'], 'validation')
        return train_dataset, val_dataset
    else:
        return post_process_dataset(dataset['train'], split)

def load_jsonl_dataset(path,split='train',val_size=0.01):
    files = glob.glob(path + '**/*.jsonl')+glob.glob(path+'*.jsonl')
    print(f'loading jsonl dataset from {files}')
    dataset = load_dataset('json', data_files=files)
    if split == 'train':
        dataset = dataset['train'].train_test_split(test_size=val_size)
        train_dataset = post_process_dataset(dataset['train'], 'train')
        val_dataset = post_process_dataset(dataset['test'], 'validation')
        return train_dataset, val_dataset
    else:
        dataset = load_dataset('json', data_files=files)
        dataset = post_process_dataset(dataset,split)
    return dataset

def check_dataset_type(path):
    parquet_files = glob.glob(path + '**/*.parquet')+glob.glob(path+'*.parquet')
    json_gz_files = glob.glob(path + '**/*.jsonl.gz')+glob.glob(path+'*.jsonl.gz')
    jsonl_files = glob.glob(path + '**/*.jsonl')+glob.glob(path+'*.jsonl')
    types = []
    if parquet_files and len(parquet_files) > 0:
        types.append('parquet')
    if json_gz_files and len(json_gz_files) > 0 :
        types.append('json_gz')
    if jsonl_files and len(jsonl_files) > 0:
        types.append('jsonl')
    return types

def post_process_dataset(dataset, split):
    # 只保留 'text' 列
    dataset = dataset.map(lambda x: {'text': x['text']}, batched=True, batch_size=1000, num_proc=10, remove_columns=dataset.column_names)
    return dataset


def load_and_interleave_datasets(paths, split='train', val_size=0.01):
    train_datasets = []
    val_datasets = []
    for path in paths:
        data_types = check_dataset_type(path)
        print(f'数据集类型：{data_types}，路径：{path}')
        if 'parquet' in data_types:
            train_dataset, val_dataset = load_parquet_dataset(path, split=split, val_size=val_size)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
        if 'json_gz' in data_types:
            train_dataset, val_dataset   = load_json_gz_dataset(path,split=split,val_size=val_size)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
        if 'jsonl' in data_types:
            train_dataset, val_dataset = load_jsonl_dataset(path,split=split,val_size=val_size)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
    train_dataset = interleave_datasets(train_datasets)
    val_dataset = interleave_datasets(val_datasets)
    return train_dataset, val_dataset

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, nargs='+', required=True)
    parser.add_argument('--val_size', type=float, default=0.1)
    args = parser.parse_args()
    
    train_dataset, val_dataset = load_and_interleave_datasets(args.paths, split='train', val_size=args.val_size)
    print(f'训练集：{train_dataset}')
    print(f'验证集：{val_dataset}')
    print("训练集样本：", train_dataset[100])
    print('训练样本:',train_dataset[101])
    print('训练样本:',val_dataset[102])
    print("验证样本：", val_dataset[100])
    print('验证样本:',val_dataset[101])
    print('验证样本:',val_dataset[102])
    print(f'训练集长度：{len(train_dataset)}')
    print(f'验证集长度：{len(val_dataset)}')
