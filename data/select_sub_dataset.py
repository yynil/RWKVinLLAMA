import os
import argparse
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def process_dataset(dataset, eos_token_id, min_len, max_len, dry_run):
    def process_example(examples):
        batch_input_ids = examples['input_ids']
        batch_size = len(batch_input_ids)
        
        valid = [False] * batch_size
        for i, input_ids in enumerate(batch_input_ids):
            # 计算真实长度
            real_length = len(input_ids)
            for token in reversed(input_ids):
                if token == eos_token_id:
                    real_length -= 1
                else:
                    break
            
            valid[i] = min_len <= real_length <= max_len
        
        examples['valid'] = valid
        return examples

    processed_dataset = dataset.map(
        process_example,
        batched=True,
        batch_size=1000,
        num_proc=os.cpu_count(),
        desc="处理数据集"
    )
    
    if dry_run:
        valid_count = sum(processed_dataset['valid'])
        print(f"符合长度要求的数据数量: {valid_count}")
    else:
        filtered_dataset = processed_dataset.filter(lambda x: x['valid'], num_proc=os.cpu_count())
        return filtered_dataset

def main(args):
    # 加载数据集
    print(f'processing {args.input_dir} to {args.output_dir} with {args.min_len} to {args.max_len}')
    dataset = load_from_disk(args.input_dir)
    
    # 检查是否包含 input_ids
    if 'input_ids' not in dataset.column_names:
        raise ValueError("数据集中不包含 input_ids 列")
    
    # 加载 tokenizer 只是为了获取 eos_token_id
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    eos_token_id = tokenizer.eos_token_id
    
    # 处理数据集
    processed_dataset = process_dataset(dataset, eos_token_id, args.min_len, args.max_len, args.dry_run)
    
    if not args.dry_run:
        # 保存处理后的数据集
        processed_dataset.save_to_disk(args.output_dir)
        print(f"处理后的数据集已保存到 {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理数据集")
    parser.add_argument("--input_dir", type=str, required=True, help="输入数据集目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出数据集目录")
    parser.add_argument("--model_name", type=str, required=True, help="模型名称或路径，用于加载 tokenizer")
    parser.add_argument("--min_len", type=int, default=0, help="最小长度")
    parser.add_argument("--max_len", type=int, required=True, help="最大长度")
    parser.add_argument("--dry_run", action="store_true", help="只统计符合条件的数据数量，不保存")
    parser.add_argument("--step", type=int, default=256, help="长度区间步长")
    args = parser.parse_args()
    #create min max range
    min_max_range = [(i, min(i+args.step-1, args.max_len)) for i in range(args.min_len, args.max_len, args.step)]
    print(min_max_range)
    base_output_dir = args.output_dir   
    for start, end in min_max_range:
        print(start, end)
        args.min_len = start
        args.max_len = end  
        args.output_dir = f"{base_output_dir}_{start}_{end}" 
        main(args)
