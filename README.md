# RWKVinLLAMA

This repository is used to distill (hybrid) RWKV model with Llama. 

# Architecture

Since the teacher model is relatively large, we design a NCCL based service to run the teacher model in a separate card while other training processes just get the logits from the teacher model service through NCCL.

```mermaid
graph LR
    subgraph Training Processes
       A1[Training Process 1]
       A2[Training Process 2]
       A3[Training Process 3]
       A4[Training Process 4]
       A5[Training Process 5]
       A6[Training Process 6]
       A7[Training Process 7]
    end
    A1 -->|NCCL| C[Teacher Model Service]
    A2 -->|NCCL| C
    A3 -->|NCCL| C
    A4 -->|NCCL| C
    A5 -->|NCCL| C
    A6 -->|NCCL| C
    A7 -->|NCCL| C
```

The teacher model service is implemented in `server/teacher_server_nccl_gather.py'. 
We gather all input_ids from training processes and run inference in batch because NCCL transfer speed is much faster than the inference speed of the teacher model.