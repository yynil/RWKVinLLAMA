import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import argparse
import asyncio
import cupy as cp
import numpy as np
from cupy.cuda import nccl
import time
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()

async def async_recv(comm, recv_buffer,r):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        executor,
        lambda: comm.recv(recv_buffer.data.ptr, recv_buffer.size, nccl.NCCL_INT64, r, cp.cuda.Stream.null.ptr)
    )
    
async def async_send(comm, logits,r):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        executor,
        lambda: comm.send(logits.data_ptr(), logits.size(0)*logits.size(1)*logits.size(2), nccl.NCCL_FLOAT, r, cp.cuda.Stream.null.ptr)
    )
def handle_request_sync(model, input_ids,eos_id):
    attention_mask = torch.ne(input_ids, eos_id).to(input_ids.device)
    with torch.no_grad():
        results =  model(input_ids, attention_mask=attention_mask)
    return results.logits
async def handle_request(model, input_ids,eos_id):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: handle_request_sync(model, input_ids,eos_id)
    )
    # attention_mask = torch.ne(input_ids, eos_id).to(input_ids.device)
    # with torch.no_grad():
    #     results =  model(input_ids, attention_mask=attention_mask)
    # return results.logits



async def run(comm,r, rank,model,batch,length,eos_id):
    
    # 每个 rank 生成自己的输入数据
    print(f"Rank {rank} start receiving rank {r} data")
    recv_buffer = cp.empty((batch, length), dtype=cp.int64)
    transfer_time = 0
    inference_time = 0
    count = 0
    while True:
        start = time.time()
        comm.recv(recv_buffer.data.ptr, recv_buffer.size, nccl.NCCL_INT64, r, cp.cuda.Stream.null.ptr)
        input_ids = torch.as_tensor(recv_buffer, device=f'cuda:{rank}', dtype=torch.long)
        end = time.time()
        print(f'finish receiving for rank {r}')
        transfer_time += end-start
        # process input
        start = time.time()
        logits = await handle_request(model, input_ids,eos_id)   
        end = time.time()
        inference_time += end-start
        print(f'finish inference for rank {r}')
        start = time.time()
        comm.send(logits.data_ptr(), logits.size(0)*logits.size(1)*logits.size(2), nccl.NCCL_FLOAT, r, cp.cuda.Stream.null.ptr)
        end = time.time()
        print(f'finish sending for rank {r}')
        transfer_time += end-start
        # send output back to client rank
        count += 1
        if count % 50 == 0:
            print(f"Rank {rank} time transfer: {transfer_time/count}, time inference: {inference_time/count}")
            print(f"Rank {rank} Ratio of transfer time to inference time: {transfer_time/inference_time}")
            transfer_time = 0
            inference_time = 0
            count = 0
        

async def main(model_id,nccl_id,rank, size, fn,batch,length,eos_id):
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id).to(dtype=dtype,device=f'cuda:{rank}')
    model.eval()
    print(f'init the server with nccl_id {nccl_id}')
    cp.cuda.Device(rank).use()
    comm = nccl.NcclCommunicator(size, nccl_id, rank)
    print('Server initialized')
    # threads = []
    # from threading import Thread
    # for r in range(size):
    #     if r != rank:
    #         thread = Thread(target=fn, args=(comm,r, rank,model,batch,length,eos_id))
    #         thread.start()
    #         threads.append(thread)
    # for thread in threads:
    #     thread.join()
    tasks = []
    for r in range(size):
        if r != rank:
            task = asyncio.create_task(fn(comm,r, rank,model,batch,length,eos_id))
            tasks.append(task)
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='model id')
    parser.add_argument('--batch', type=int, required=True, help='batch size')
    parser.add_argument('--length', type=int, required=True, help='length of input')
    parser.add_argument('--size', type=int, required=True, help='number of nodes')
    parser.add_argument('--nccl_id_file', type=str, default='nccl.txt',help='nccl id file')
    parser.add_argument('--rank', type=int, required=False, default=3, help='rank of the current process')
    args = parser.parse_args()
    nccl_id_file = args.nccl_id_file
    nccl_id = nccl.get_unique_id()
    print("NCCL ID:", nccl_id)
    #save nccl_id to file with json format
    with open(nccl_id_file, 'w') as f:
        import json
        json.dump({'nccl_id':nccl_id},f)
    size = args.size
    batch = args.batch
    length = args.length
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    eos_id = tokenizer.eos_token_id
    rank = args.rank
    asyncio.run(main(args.model_id, nccl_id,rank,size,run,batch,length,eos_id))