import cupy as cp
import numpy as np
from cupy.cuda import nccl
import os
import multiprocessing as mp

def server_process(nccl_id):
    # 设置 GPU 设备
    cp.cuda.Device(0).use()

    # 初始化 NCCL 通信器
    print("Initializing NCCL communicator")
    comm = nccl.NcclCommunicator(2, nccl_id, 0)
    print("Server initialized")

    # Server 逻辑
    for _ in range(5):  # 执行 5 次数据交换
        # 准备发送数据
        send_data = cp.array([1.0, 2.0, 3.0])
        comm.send(send_data, 1)
        print("Server sent:", send_data)

        # 接收数据
        recv_data = cp.zeros_like(send_data)
        comm.recv(recv_data, 1)
        print("Server received:", recv_data)

    print("Server finished")

if __name__ == "__main__":
    # 生成唯一的 NCCL ID
    import sys
    import struct
    nccl_id_file = sys.argv[1]
    nccl_id = nccl.get_unique_id()
    print("NCCL ID:", nccl_id)
    #save nccl_id to file with json format
    with open(nccl_id_file, 'w') as f:
        import json
        json.dump({'nccl_id':nccl_id},f)
    server_process(nccl_id)
