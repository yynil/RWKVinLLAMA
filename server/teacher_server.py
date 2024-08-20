import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM,AutoTokenizer
import argparse
import asyncio

async def handle_request(rank, model, input_ids,eos_id):
    print(f"Rank {rank} processing input")
    with torch.no_grad():
        attention_mask = torch.ne(input_ids, eos_id).to(input_ids.device)
        results = model(input_ids, attention_mask=attention_mask)
        logits = results.logits
        await asyncio.sleep(0)
        return logits

async def run(rank, size,model,batch,length,eos_id):
    
    # 每个 rank 生成自己的输入数据
    input_ids = torch.zeros((batch, length), dtype=torch.long).to(0)
    
    while True:
        print(f"Rank {rank} waiting for input")
        
        
        # receive input from client rank
        dist.recv(input_ids, src=rank)
        # process input
        logits = await handle_request(rank, model, input_ids,eos_id)
        # send output back to client rank
        print(f'Rank {rank} sending logits,shape is {logits.shape}')
        dist.send(logits, dst=rank)
        del logits
        torch.cuda.empty_cache()
        

async def main(model_id, size, master_addr, master_port, fn,batch,length,eos_id, backend='nccl'):
    dist.init_process_group(backend, rank=0, world_size=size, init_method=f'tcp://{master_addr}:{master_port}',group_name='teacher_name')
    
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(0)
    model.eval()
    tasks = []
    for rank in range(1,size):
        task = asyncio.create_task(run(rank, size,model,batch,length,eos_id))
        tasks.append(task)
        
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='model id')
    parser.add_argument('--master_addr', type=str, required=True, help='address of the master node')
    parser.add_argument('--master_port', type=str, required=True, help='port of the master node')
    parser.add_argument('--batch', type=int, required=True, help='batch size')
    parser.add_argument('--length', type=int, required=True, help='length of input')
    parser.add_argument('--size', type=int, required=True, help='number of nodes')
    args = parser.parse_args()
    size = args.size
    batch = args.batch
    length = args.length
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    eos_id = tokenizer.eos_token_id
    asyncio.run(main(args.model_id, size, args.master_addr, args.master_port, run,batch,length,eos_id))