import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import argparse
import asyncio
import cupy as cp
import numpy as np
from cupy.cuda import nccl
import time
from concurrent.futures import ThreadPoolExecutor
import logging
import os
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

def handle_request_sync(model, input_ids,eos_id,output_all_hiddens=False):
    logging.info(f"Start inference,input_ids shape is {input_ids.shape}, eos_id is {eos_id} input_ids {input_ids},output_all_hiddens is {output_all_hiddens}")  
    with torch.no_grad():
        results =  model(input_ids,output_hidden_states=output_all_hiddens)
    logging.info(f"Finished inference,result logits shape is {results.logits.shape}")
    return results.logits,results.hidden_states

        

def main(model_id,nccl_id,device_id, size, batch,length,eos_id,output_all_hiddens=False):
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id).to(dtype=dtype,device=f'cuda:{device_id}')
    layers = model.config.num_hidden_layers
    model.eval()
    logging.info(f'init the server with nccl_id {nccl_id}')
    cp.cuda.Device(device_id).use()
    comm = nccl.NcclCommunicator(size, nccl_id, size-1)
    logging.info('Server initialized')
    recv_buffer = cp.empty(((size-1)*batch, length), dtype=cp.int64)
    #receive data from other ranks, run in thread pool
    transfer_time = 0
    inference_time = 0
    count = 0
    stream = cp.cuda.Stream(non_blocking=True)
    while True:
        start = time.time()
        # futures = []
        for i in range(size-1):
            logging.info(f"Rank {size-1} start receiving rank {i} data")
            rank_recv_buffer = recv_buffer.data.ptr+i*batch*length*8
            comm.recv(rank_recv_buffer, batch*length, nccl.NCCL_INT64, i, stream.ptr)
            logging.info(f"Rank {size-1} finished receiving rank {i} data")    
        stream.synchronize()

        end = time.time()
        transfer_time += end-start
        input_ids = torch.as_tensor(recv_buffer, device=f'cuda:{device_id}', dtype=torch.long)
        logging.info(f"Rank {size-1} received input_ids, shape is {input_ids.shape} input_ids dtype is {input_ids.dtype}")
        # logging.info(f'input_ids is {input_ids}')
        start = time.time()
        logits,hidden_states = handle_request_sync(model, input_ids,eos_id,output_all_hiddens=output_all_hiddens)
        end = time.time()
        inference_time += end-start
        start = time.time()
        # logging.info(f"Rank {size-1} inference time is {end-start}")
        # futures = []
        for i in range(size-1):
            rank_logits = logits[i*batch:(i+1)*batch]#(batch,length,vocab_size)z
            # logging.info(f"Rank {size-1} sending logits to rank {i} rank_logits shape is {rank_logits.shape}")
            # logging.info(f'RANK{i} rank_logits[0]: {rank_logits[0]}')
            rank_data_ptr = rank_logits.data_ptr()
            comm.send(rank_data_ptr, rank_logits.size(0)*rank_logits.size(1)*rank_logits.size(2), nccl.NCCL_FLOAT, i, stream.ptr)
            if hidden_states is not None and output_all_hiddens:
                # logging.info(f"all length of hidden_states is {len(hidden_states)}")
                rank_hidden_states = [hidden_states[num_layer][i*batch:(i+1)*batch]  for num_layer in range(1,layers+1)]
                # logging.info(f"Rank {size-1} sending hidden_states to rank {i}")
                #Since hiddens are list of tensors, we concatenate them to a single tensor and send them back
                rank_hidden_states = torch.cat(rank_hidden_states,dim=0)#num_layers*batch,length,hidden_size
                logging.info(f"Rank {size-1} sending hidden_states to rank {i} rank_hidden_states shape is {rank_hidden_states.shape}")
                rank_hidden_states = rank_hidden_states.to(torch.float32)
                rank_data_ptr = rank_hidden_states.data_ptr()
                logging.info(f"Rank {size-1} sending hidden_states to rank {i} rank_hidden_states shape is {rank_hidden_states.shape}")
                comm.send(rank_data_ptr, rank_hidden_states.size(0)*rank_hidden_states.size(1)*rank_hidden_states.size(2), nccl.NCCL_FLOAT, i, stream.ptr)
        stream.synchronize()
        end = time.time()
        transfer_time += end-start
        del logits
        del hidden_states
        del input_ids
        torch.cuda.empty_cache()
        count += 1
        if count % 50 == 0:
            logging.info(f"Rank {size-1} time transfer: {transfer_time/count}, time inference: {inference_time/count}")
            logging.info(f"Rank {size-1} Ratio of transfer time to inference time: {transfer_time/inference_time}")
            transfer_time = 0
            inference_time = 0
            count = 0
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='model id')
    parser.add_argument('--batch', type=int, required=True, help='batch size')
    parser.add_argument('--length', type=int, required=True, help='length of input')
    parser.add_argument('--size', type=int, required=True, help='number of nodes')
    parser.add_argument('--output_all_hiddens', action='store_true',default=False, help='return all hiddens')
    parser.add_argument('--nccl_id_file', type=str, default='nccl.txt',help='nccl id file')
    parser.add_argument('--device_id', type=int, default=0, help='device id')
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
    device_id = args.device_id
    main(args.model_id, nccl_id,device_id,size,batch,length,eos_id,output_all_hiddens=args.output_all_hiddens)