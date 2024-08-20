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


def handle_request_sync(model, input_ids,eos_id):
    attention_mask = torch.ne(input_ids, eos_id).to(input_ids.device)
    with torch.no_grad():
        results =  model(input_ids, attention_mask=attention_mask)
    return results.logits

        

def main(model_id,nccl_id, size, batch,length,eos_id):
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id).to(dtype=dtype,device=f'cuda:{size-1}')
    model.eval()
    print(f'init the server with nccl_id {nccl_id}')
    cp.cuda.Device(size-1).use()
    comm = nccl.NcclCommunicator(size, nccl_id, size-1)
    print('Server initialized')
    recv_buffer = cp.empty(((size-1)*batch, length), dtype=cp.int64)
    #receive data from other ranks, run in thread pool
    transfer_time = 0
    inference_time = 0
    count = 0
    while True:
        start = time.time()
        # futures = []
        for i in range(size-1):
            # print(f"Rank {size-1} start receiving rank {i} data")
            rank_recv_buffer = recv_buffer.data.ptr+i*batch*length*8
            comm.recv(rank_recv_buffer, batch*length, nccl.NCCL_INT64, i, cp.cuda.Stream.null.ptr)
            # furture = executor.submit(comm.recv, rank_recv_buffer, batch*length, nccl.NCCL_INT64, i, cp.cuda.Stream.null.ptr)
            # futures.append(furture)
        # for future in futures:
            # future.result()
        end = time.time()
        transfer_time += end-start
        input_ids = torch.as_tensor(recv_buffer, device=f'cuda:{size-1}', dtype=torch.long)
        # print(f"Rank {size-1} received input_ids, shape is {input_ids.shape} input_ids dtype is {input_ids.dtype}")
        # print(f'input_ids is {input_ids}')
        start = time.time()
        logits = handle_request_sync(model, input_ids,eos_id)
        end = time.time()
        inference_time += end-start
        start = time.time()
        # futures = []
        for i in range(size-1):
            rank_logits = logits[i*batch:(i+1)*batch]
            # print(f"Rank {size-1} sending logits to rank {i} rank_logits shape is {rank_logits.shape}")
            # print(f'RANK{i} rank_logits[0]: {rank_logits[0]}')
            rank_data_ptr = rank_logits.data_ptr()
            comm.send(rank_data_ptr, rank_logits.size(0)*rank_logits.size(1)*rank_logits.size(2), nccl.NCCL_FLOAT, i, cp.cuda.Stream.null.ptr)
            # future = executor.submit(comm.send, rank_data_ptr, rank_logits.size(0)*rank_logits.size(1)*rank_logits.size(2), nccl.NCCL_FLOAT, i, cp.cuda.Stream.null.ptr)
            # futures.append(future)
        # for future in futures:
            # future.result()
        end = time.time()
        transfer_time += end-start
        count += 1
        if count % 50 == 0:
            print(f"Rank {size-1} time transfer: {transfer_time/count}, time inference: {inference_time/count}")
            print(f"Rank {size-1} Ratio of transfer time to inference time: {transfer_time/inference_time}")
            transfer_time = 0
            inference_time = 0
            count = 0
    
    # threads = []
    # from threading import Thread
    # for r in range(size):
    #     if r != rank:
    #         thread = Thread(target=fn, args=(comm,r, rank,model,batch,length,eos_id))
    #         thread.start()
    #         threads.append(thread)
    # for thread in threads:
    # #     thread.join()
    # tasks = []
    # for r in range(size):
    #     if r != rank:
    #         task = asyncio.create_task(fn(comm,r, rank,model,batch,length,eos_id))
    #         tasks.append(task)
    # await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='model id')
    parser.add_argument('--batch', type=int, required=True, help='batch size')
    parser.add_argument('--length', type=int, required=True, help='length of input')
    parser.add_argument('--size', type=int, required=True, help='number of nodes')
    parser.add_argument('--nccl_id_file', type=str, default='nccl.txt',help='nccl id file')
    args = parser.parse_args()
    
    nccl_id_file = args.nccl_id_file
    size = args.size
    #save nccl_id to file with json format
    nccl_id = nccl.get_unique_id()
    with open(nccl_id_file, 'w') as f:
        import json
        json.dump({'nccl_id':nccl_id},f)

    batch = args.batch
    length = args.length
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    eos_id = tokenizer.eos_token_id
    main(args.model_id, nccl_id,size,batch,length,eos_id)