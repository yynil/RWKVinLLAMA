import sys
import os

# 获取要添加的目录路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 打印要添加的目录
print(f"正在添加以下路径到 sys.path: {project_root}")

# 添加项目根目录到 sys.path
sys.path.append(project_root)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rwkv_llama.utilities import HybridCache

# 检查CUDA是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 设置模型路径
model_id = "/home/yueyulin/model/llama-3.1-8B-Instruct/"
model_id = '/data/rwkv/models/meta-llama/Meta-Llama-3.1-8B-Instruct/'
input_text = "In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
input_text = "Which number is greater, 9.11 or 9.8?"
def main(model_id, input_text):
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # 使用FP16精度
    device_map="auto"  # 自动选择可用的GPU
    )

    # 创建HybridCache实例
    cache = HybridCache()

# 准备输入

    conversations = [
        {
            'role': 'user',
        'content': input_text
        }
    ]
    input_text = tokenizer.apply_chat_template(conversations,tokenize=False,add_generation_prompt=True)
    print(input_text)
    input_ids = tokenizer(input_text, return_tensors="pt").to(device)
    print(input_ids)

    # 使用模型生成输出,同时使用HybridCache
    with torch.no_grad():
        output = model.generate(
        input_ids = input_ids['input_ids'],
        attention_mask = input_ids['attention_mask'],
        max_length=500,
        num_return_sequences=1,
        past_key_values=cache,
        use_cache=True,
        do_sample=False,
        num_beams=1,
    )

    # 解码输出
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("生成的文本:")
    print(generated_text)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default=model_id)
    parser.add_argument('--input_text', type=str, default=input_text)
    args = parser.parse_args()
    main(args.model_id, args.input_text)
