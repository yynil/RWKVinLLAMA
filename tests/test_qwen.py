model_id = '/home/yueyulin/models/Qwen2.5-7B-Instruct/'
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(tokenizer)
llama_model_id = '/home/yueyulin/models/llama-3.1-8B-Instruct/'
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id) 

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(text)

llama_text = llama_tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(llama_text)
print(tokenizer.eos_token_id)