import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel
import argparse
import time
import cupy as cp
from cupy.cuda import nccl

def init_process(nccl_id,rank,server_rank, size, fn,batch,length,vocab_size,device_id):
    cp.cuda.Device(device_id).use()
    print(f"Rank {rank} initializing process")
    comm = nccl.NcclCommunicator(size, nccl_id, rank)
    print(f"Rank {rank} initialized communicator")
    time.sleep(1)
    fn(comm,rank,server_rank, batch,length,device_id,vocab_size)

def run(comm,rank,server_rank, batch,length,device_id,vocab_size=128256):
    recv_buffer = cp.empty((batch, length,vocab_size), dtype=cp.float32)
    while True:
        # print(f"Rank {rank} sending for input")
        input_ids = torch.randint(0, 128255, (batch, length), dtype=torch.long).to(rank)
        # print(input_ids)
        comm.send(input_ids.data_ptr(), input_ids.size(0)*input_ids.size(1), nccl.NCCL_INT64, server_rank, cp.cuda.Stream.null.ptr)
        comm.recv(recv_buffer.data.ptr, recv_buffer.size, nccl.NCCL_FLOAT, server_rank, cp.cuda.Stream.null.ptr)
        logits = torch.as_tensor(recv_buffer, device=f'cuda:{device_id}', dtype=torch.float32)
        # print(f"Rank {rank} received logits, shape is {logits.shape} logits dtype is {logits.dtype}")
        # print(f"logits[0]: {logits[0]}")
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True, help='rank of the current process')
    parser.add_argument('--batch', type=int, required=True, help='batch size')
    parser.add_argument('--length', type=int, required=True, help='length of input')
    parser.add_argument('--size', type=int, required=True, help='number of nodes')
    parser.add_argument('--vocab_size', type=int, required=False, default=128256, help='vocab size')
    parser.add_argument('--nccl_id_file', type=str,  default='nccl.txt',help='nccl id file')
    parser.add_argument('--server_rank', type=int, required=False, default=3, help='rank of the server process')
    parser.add_argument('--device_id', type=int, help='device id')
    args = parser.parse_args()
    size = args.size
    batch = args.batch
    length = args.length
    vocab_size=args.vocab_size
    nccl_id_file = args.nccl_id_file
    server_rank = args.server_rank
    with open(nccl_id_file, 'r') as f:
        import json
        nccl_id = json.load(f)['nccl_id']
        nccl_id = tuple(nccl_id)
    print("NCCL ID:", nccl_id)
    init_process(nccl_id,args.rank,server_rank, size, run,batch,length,vocab_size,args.device_id)