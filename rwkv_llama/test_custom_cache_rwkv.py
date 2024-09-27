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
config_file = "configs/test_hybrid_full_logits_llamamlp_local.yaml"
config_file = "configs/step_wise/test_hybrid_1_layer_llamamlp.yaml"
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
ckpt_file = '/home/yueyulin/model/all_llama3_1.pth'
ckpt_file = '/data/rwkv/tmp/distill-en-zh_llama3_1_pseudo_ds_all_kl_div_stepwise_layer_1/one_layer.pth'
model.load_ckpt(ckpt_file)
# 创建HybridCache实例
cache = HybridCache()

# 准备输入
# input_text = "In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
# input_text = "你是一个旅游行程规划专家，请根据用户需求生成他们需要的旅行计划，要尽可能详细到时间和地点。\n\nAssistant: 好的。\n\nUser: 请为我生成一份在北京两天的旅游计划。\n\nAssistant: 好的"
input_text = "User: 请生成一个 Python 程序,输入一个整数列表,找到该列表的最大值，并且打印出最大值和最大值的位置。\n\nAssistant: 好的"
input_text = "Please prepare a journey plan for me for two days in Beijing."
input_text = "Tell me something about Alexander the Great."
# input_text = "请告诉我一些关于亚历山大大帝的信息。"
input_text = "Why is the sky blue?"
input_text = "Does programming language Rust is faster than Python? Why?"
input_text = "Which number is greater, 9.11 or 9.8?"
# input_text = "Please generate a Python program that takes an integer list as input, finds the maximum value of the list, and prints out the maximum value and the position of the maximum value."
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
    
model = model.to(dtype=torch.bfloat16,device='cuda')
model.eval()
# 使用模型生成输出,同时使用HybridCache
with torch.no_grad():
    output = model.model.generate(
        input_ids = input_ids['input_ids'],
        attention_mask = input_ids['attention_mask'],
        max_length=2048,
        num_return_sequences=1,
        past_key_values=cache,
        use_cache=True,stop_strings=["<|eot_id|>"], tokenizer=tokenizer
    )

# 解码输出
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("生成的文本:")
print(generated_text)
