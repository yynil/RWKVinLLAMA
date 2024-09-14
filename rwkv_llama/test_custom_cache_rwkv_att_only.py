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
from transformers import AutoModelForCausalLM, AutoTokenizer
config_file = "configs/test_hybrid_full_logits_rwkv_att_only.yaml"
import yaml
with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print(config)
model_id = config['Llama']['model_id']
tokenizer = AutoTokenizer.from_pretrained(model_id)
transformer_model = AutoModelForCausalLM.from_pretrained(model_id)
print(transformer_model)
from hybrid_model_run import create_rwkv_args,HybridModel
args = create_rwkv_args(transformer_model.config, config)
print(args)
model = HybridModel(transformer_model,args)
print(model)
ckpt_file = '/home/yueyulin/model/steps_12k_0_256.pth'
model.load_ckpt(ckpt_file)
# 创建HybridCache实例
cache = HybridCache()

# 准备输入
# input_text = "In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
input_text = "User: 请为我生成一份在北京两天的旅游计划。\n\nAssistant: 好的"
# input_text = "User: 请生成一个 Python 程序,输入一个整数列表,找到该列表的最大值，并且打印出最大值和最大值的位置。\n\nAssistant: 好的"
# input_text = "User: Please prepare a journey plan for me for two days in Beijing.\n\nAssistant: Okay."
# input_text = "User: Please generate a Python program that takes an integer list as input, finds the maximum value of the list, the return value should be a tuple which is the max value and the index of the max value in the original list.\n\nAssistant: Okay."
input_ids = tokenizer(input_text, return_tensors="pt").to(device)
print(input_ids)
    
model = model.to(dtype=torch.bfloat16,device='cuda')
model.eval()
# 使用模型生成输出,同时使用HybridCache
with torch.no_grad():
    output = model.model.generate(
        input_ids = input_ids['input_ids'],
        attention_mask = input_ids['attention_mask'],
        max_length=500,
        num_return_sequences=1,
        past_key_values=cache,
        use_cache=True,stop_strings=["\n\nUser:"], tokenizer=tokenizer
    )

# 解码输出
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("生成的文本:")
print(generated_text)
