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
config_file = "configs/test_hybrid_full_logits_stage_2.yaml"
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
model = HybridModel(transformer_model,args)
print(model)
ckpt_file = '/home/yueyulin/model/hybrid/hybrid_model_512_15k.pt'
model.load_ckpt(ckpt_file)
# 创建HybridCache实例
cache = HybridCache()

# 准备输入
input_text = "In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
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
        use_cache=True
    )

# 解码输出
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("生成的文本:")
print(generated_text)
