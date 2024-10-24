import os
import sys

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(f'add {project_root} to sys.path')

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from rwkv_llama.utilities import HybridCache
from rwkv_llama.hybrid_model_run import create_rwkv_args, HybridModel
from prompt_chinese import PROMPTS
from cachetools import cached, TTLCache, LRUCache
from typing import List
from transformers import StoppingCriteria, StoppingCriteriaList,StopStringCriteria
model = None
tokenizer = None
#################PRORFILING#################
from torch.profiler import profile,record_function,ProfilerActivity
############################################


def load_model(config_file, ckpt_file,device):
    global model, tokenizer
    
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model_id = config['Llama']['model_id']
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    transformer_model = AutoModelForCausalLM.from_pretrained(model_id,attn_implementation="flash_attention_2")
    
    args = create_rwkv_args(transformer_model.config, config)
    model = HybridModel(transformer_model, args)
    model.load_ckpt(ckpt_file)
    model = model.to(dtype=torch.bfloat16, device=device)
    model.eval()
    # 使用 torch.compile() 编译模型
    model = torch.compile(model)
    print(model)    
    return "模型加载成功!"
def get_cache_key(prompt: str,model : HybridModel,tokenizer: AutoTokenizer,device: str): 
    return prompt
@cached(cache=TTLCache(maxsize=1000, ttl=60*5),key=get_cache_key)
def get_hybrid_cache(prompt: str,model : HybridModel,tokenizer: AutoTokenizer,device: str):
    conversations = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    prompt = tokenizer.apply_chat_template(conversations,tokenize=False,add_generation_prompt=False)
    prompt = prompt[:-len("<|im_end|>")-1]
    print(prompt)
    input_ids = tokenizer(prompt,return_tensors='pt')
    attention_mask = input_ids.attention_mask
    input_ids = input_ids.input_ids
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    cache = HybridCache()
    with torch.no_grad():
        model.model.forward(input_ids=input_ids,attention_mask=attention_mask,past_key_values=cache,use_cache=True,return_dict=True) 
    # model.model.generate(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         max_new_tokens=1,[
    #         num_return_sequences=1,
    #         past_key_values=cache,
    #         use_cache=True,
    #         early_stopping=True,
    #         do_sample=True,
    #     )
    return cache

def generate_text(prompt: str,history: List[str], model : HybridModel,tokenizer: AutoTokenizer,cache: HybridCache, device: str,stop_text: str,profile_enabled: bool=False):
    history.append({"role": "user", "content": prompt})
    
    prompt = tokenizer.apply_chat_template(history,tokenize=False,add_generation_prompt=True)
    print(prompt)
    input_ids = tokenizer(prompt,return_tensors='pt')
    attention_mask = input_ids.attention_mask
    input_ids = input_ids.input_ids
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    input_length = input_ids.shape[1]
    stopping_criteria = StoppingCriteriaList([StopStringCriteria(tokenizer,[stop_text])]) if stop_text else None
    with torch.no_grad():
        if profile_enabled:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
                output = model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2048,
                num_return_sequences=1,
                past_key_values=cache,
                use_cache=True,
                do_sample=False,
                early_stopping=True,
                stopping_criteria=stopping_criteria
            )
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        else:
            output = model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2048,
                num_return_sequences=1,
                past_key_values=cache,
                use_cache=True,
                early_stopping=True,
                stopping_criteria=stopping_criteria,
                do_sample=False
            )
    generated_text = tokenizer.decode(output[0,input_length:], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    prefix_template = PROMPTS["entiy_extraction_prefix"]
    print(prefix_template)
    context_base = dict(    
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types="person, role, organization, location, opinion, concept, date",
        output_language="中文",
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]
    input_text = """
    贷款市场报价利率（Loan Prime Rate, LPR）是由具有代表性的报价行，根据本行对最优质客户的贷款利率，以公开市场操作利率（主要指中期借贷便利利率）加点形成的方式报价，由中国人民银行授权全国银行间同业拆借中心计算并公布的基础性的贷款参考利率，各金融机构应主要参考LPR进行贷款定价。 现行的LPR包括1年期和5年期以上两个品种 [1]。LPR市场化程度较高，能够充分反映信贷市场资金供求情况，使用LPR进行贷款定价可以促进形成市场化的贷款利率，提高市场利率向信贷利率的传导效率。
    2020年8月12日，中国工商银行、中国建设银行、中国农业银行、中国银行和中国邮政储蓄银行五家国有大行同时发布公告，于8月25日起对批量转换范围内的个人住房贷款，按照相关规则统一调整为LPR（贷款市场报价利率）定价方式。 [2]
    最新贷款市场报价利率（LPR）：2024年10月21日，1年期LPR为3.10%，5年期以上LPR为3.60%，均较此前下降0.25个百分点。
    """

    #--config_file configs/step_wise/test_hybrid_5_layer_qwenmlp_local.yaml --ckpt_file /home/yueyulin/model/qwen/layer5.pth
    config_file = 'configs/step_wise/test_hybrid_5_layer_qwenmlp_local.yaml'
    ckpt_file = '/home/yueyulin/model/qwen/layer5.pth'
    device = 'cuda:0'
    
    prefix = prefix_template.format(**context_base)
    print(prefix)
    
    real_data_template = PROMPTS["real_data"]
    real_data = real_data_template.format(input_text=input_text,**context_base)
    print(real_data)
    load_model(config_file,ckpt_file,device)
    history = []
    stop_text = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
    cache = get_hybrid_cache(prefix,model,tokenizer,device)
    print(cache)
    all_data = prefix + real_data
    generated_text = generate_text(all_data,history,model,tokenizer,cache,device,stop_text)
    print(generated_text)
    print(cache)
    history.append({"role": "assistant", "content": generated_text})
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    generated_text = generate_text(continue_prompt,history,model,tokenizer,cache,device,None)
    print(generated_text)
    print(cache)
    history.append({"role": "assistant", "content": generated_text})    
    confirm_prompt = PROMPTS["entiti_if_loop_extraction"]
    generated_text = generate_text(confirm_prompt,history,model,tokenizer,cache,device,None)
    print(generated_text)
    print(cache)