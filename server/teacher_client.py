import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel
import argparse
import time
def init_process(rank, size, master_addr, master_port, fn,batch,length,vocab_size ,backend='nccl'):
    dist.init_process_group(backend, rank=rank, world_size=size, init_method=f'tcp://{master_addr}:{master_port}',group_name='teacher_name')
    fn(rank, size,batch,length,vocab_size)

def run(rank, size,batch,length,vocab_size=128256):
    while True:
        print(f"Rank {rank} sending for input")
        input_ids = torch.randint(100, 10000, (batch, length), dtype=torch.long).to(rank)
        dist.send(input_ids, dst=0)
        logits = torch.zeros((batch, length, vocab_size), dtype=torch.float).to(rank)
        dist.recv(logits, src=0)
        print(f"Rank {rank} received logits shape: {logits.shape}")
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_addr', type=str, required=True, help='address of the master node')
    parser.add_argument('--master_port', type=str, required=True, help='port of the master node')
    parser.add_argument('--rank', type=int, required=True, help='rank of the current process')
    parser.add_argument('--batch', type=int, required=True, help='batch size')
    parser.add_argument('--length', type=int, required=True, help='length of input')
    parser.add_argument('--size', type=int, required=True, help='number of nodes')
    parser.add_argument('--vocab_size', type=int, required=False, default=128256, help='vocab size')
    args = parser.parse_args()
    size = args.size
    batch = args.batch
    length = args.length
    vocab_size=args.vocab_size
    init_process(args.rank, size, args.master_addr, args.master_port, run,batch,length,vocab_size)