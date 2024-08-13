import ctypes
import yaml
from multiprocessing import shared_memory
import time
from transformers import AutoModelForCausalLM
import torch
import numpy
from multiprocessing import Process, resource_tracker
from multiprocessing.shared_memory import SharedMemory
import time
import ray
infere_time = 0.0
cpy_mem_time = 0.0
infere_count = 0

@ray.remote(num_gpus=1)
class TeacherServer:
    def __init__(self, model_id,device,dtype):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        print(f'Initializing TeacherServer with model {model_id} on {device} with dtype {dtype}')
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map={'':'cpu'})
        model = model.to(device)
        self.model = model
        self.model.eval()
        print(f'Initialized TeacherServer with model {model}')
    def haha(self):
        print('haha')
        return 1
    
    def compute_logits(self,input_ids:torch.Tensor)->torch.Tensor:
        global infere_time
        global cpy_mem_time
        global infere_count
        start = time.time()
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = torch.ne(input_ids, 0)
            logits = self.model(input_ids,attention_mask=attention_mask).logits
        infere_time += time.time()-start
        infere_count += 1
        return logits
if __name__ == "__main__":
    # remove_shm_from_resource_tracker()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/teacher_server.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
    model_id = config['model_id']
    device = config['device']
    dtype = config['dtype']
    if dtype == 'bfloat16':
        dtype = torch.bfloat16
    elif dtype == 'float16':
        dtype = torch.float16
    elif dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError(f'Invalid dtype: {dtype}')
    
    
    ray.init(namespace=config['name_space'])
    c = TeacherServer.options(name="teacher").remote(model_id=model_id,device=device,dtype=dtype)
    input("Press any key to exit...")