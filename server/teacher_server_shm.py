import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import argparse
import asyncio
import cupy as cp
import numpy as np
from cupy.cuda import nccl
from multiprocessing import shared_memory
import ctypes
import time
@torch.inference_mode()
def handle_request(rank, model, input_ids,eos_id):
    attention_mask = torch.ne(input_ids, eos_id).to(input_ids.device)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        results = model(input_ids, attention_mask=attention_mask)
    logits = results.logits
    return logits
def run(shared_memories,rank,model,batch,length,eos_id):
    print(f"Rank {rank} initializing process ")
    time_inference = 0
    time_transfer = 0
    count = 0
    while True:
        while shared_memories[0].buf[0] == 0:
            time.sleep(0.01)
        start = time.time()
        input_ids = torch.frombuffer(shared_memories[1].buf, dtype=torch.long).view(batch,length)
        input_ids = input_ids.to('cuda:0')
        end = time.time()
        time_transfer += end-start
        # print(input_ids.shape)
        # print(input_ids)
        # process input
        start = time.time()
        logits = handle_request(rank, model, input_ids,eos_id)        
        end = time.time()
        time_inference += end-start
        #copy logits to shared_memory[2] buffer
        start = time.time()
        logits = logits.cpu()
        data_ptr =(ctypes.c_ubyte*batch*length*model.config.vocab_size*4).from_address(logits.data_ptr())
        shared_memories[2].buf[:] = bytearray(data_ptr)
        end = time.time()
        time_transfer += end-start
        # print(logits[0])
        # send output back to client rank
        del logits
        shared_memories[0].buf[0] = 0
        count += 1
        if count % 100 == 0:
            print(f"Rank {rank} time transfer: {time_transfer/count}, time inference: {time_inference/count}")
            print(f"Rank {rank} Ratio of transfer time to inference time: {time_transfer/time_inference}")
            time_inference = 0
            time_transfer = 0
            count = 0
        
def main(model_id, size, fn,batch,length,eos_id):
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id).to(dtype=dtype,device='cuda:0')
    model.eval()
    #initiate the shared memory for other nodes
    nodes_shm = []
    for i in range(size):
        print(f'init the server with rank {i}')
        ready = shared_memory.SharedMemory(create=True, size=1,name=f'ready_{i}')
        ready.buf[0] = 0    
        input_ids_shm = shared_memory.SharedMemory(create=True, size=batch*length*8,name=f'input_ids_{i}')
        response_shm = shared_memory.SharedMemory(create=True, size=batch*length*model.config.vocab_size*4,name=f'response_{i}')
        nodes_shm.append((ready,input_ids_shm,response_shm))
    print('Server initialized')
    threads = []
    from threading import Thread
    for i in range(size):
        thread = Thread(target=fn, args=(nodes_shm[i],i,model,batch,length,eos_id))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='model id')
    parser.add_argument('--batch', type=int, required=True, help='batch size')
    parser.add_argument('--length', type=int, required=True, help='length of input')
    parser.add_argument('--size', type=int, required=True, help='number of nodes')
    args = parser.parse_args()
    size = args.size
    batch = args.batch
    length = args.length
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    eos_id = tokenizer.eos_token_id
    main(args.model_id,size,run,batch,length,eos_id)