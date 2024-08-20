import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel
import argparse
import time
import ctypes
from multiprocessing import shared_memory
def init_process(rank, fn,batch,length,vocab_size):
    ready = shared_memory.SharedMemory(create=False, size=1,name=f'ready_{rank}')
    input_ids_shm = shared_memory.SharedMemory(create=False, size=batch*length*8,name=f'input_ids_{rank}')
    response_shm = shared_memory.SharedMemory(create=False, size=batch*length*vocab_size*4,name=f'response_{rank}')
    fn(rank,ready,input_ids_shm,response_shm,batch,length,vocab_size)

def run(rank,ready,input_ids_shm,response_shm,batch,length,vocab_size=128256):
    while True:
        print(f"Rank {rank} sending for input")
        input_ids = torch.randint(100, 10000, (batch, length), dtype=torch.long).to('cpu')
        #copy input_ids to shared_memory[1] buffer with original bytes order and format
        # 将数据指针转换为 ctypes 数组
        data_size = batch*length*8
        data_array = (ctypes.c_ubyte*data_size).from_address(input_ids.data_ptr())
        input_ids_shm.buf[:batch*length*8] =  bytearray(data_array)
        ready.buf[0] = 1
        print(input_ids)
        while ready.buf[0] == 1:
            time.sleep(0.0001)
        logits = torch.frombuffer(response_shm.buf, dtype=torch.float32).view(batch,length,vocab_size)
        print(f"Rank {rank} received logits shape: {logits.shape}")
        print(logits[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True, help='rank of the current process')
    parser.add_argument('--batch', type=int, required=True, help='batch size')
    parser.add_argument('--length', type=int, required=True, help='length of input')
    parser.add_argument('--vocab_size', type=int, required=False, default=128256, help='vocab size')
    args = parser.parse_args()
    batch = args.batch
    length = args.length
    vocab_size=args.vocab_size
    init_process(args.rank, run,batch,length,vocab_size)