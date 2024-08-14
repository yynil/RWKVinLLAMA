import cupy as cp
import numpy as np
from cupy.cuda import nccl
import os
import multiprocessing as mp
def enable_p2p():
    num_gpus = cp.cuda.runtime.getDeviceCount()
    for i in range(num_gpus):
        for j in range(num_gpus):
            if i != j:
                can_access_peer = cp.cuda.runtime.deviceCanAccessPeer(i, j)
                try:
                    if can_access_peer:
                        cp.cuda.runtime.deviceEnablePeerAccess(j)
                        print(f"Enabled P2P access between GPU {i} and GPU {j}")
                    else:
                        print(f"Cannot enable P2P access between GPU {i} and GPU {j}")
                except Exception as e:
                    print(f"Cannot enable P2P access between GPU {i} and GPU {j} because {e}") 
enable_p2p()
def server_process(nccl_id):
    # 设置 GPU 设备
    cp.cuda.Device(2).use()

    # 初始化 NCCL 通信器
    print("Initializing NCCL communicator")
    comm = nccl.NcclCommunicator(2, nccl_id, 0)
    print("Server initialized")
    test_send_size = 2048*4096*20
    import time
    # Server 逻辑
    for i in range(5):  # 执行 5 次数据交换
        # 准备发送数据
        start_time = time.time()
        send_data = cp.array([2.0]*test_send_size,dtype=cp.float32)
        comm.send(send_data.data.ptr, send_data.size, nccl.NCCL_FLOAT, 1, cp.cuda.Stream.null.ptr)
        # cp.cuda.Stream.null.synchronize()
        end_time = time.time()
        print("Server sent (iteration {i}):", send_data, "Size:", send_data.nbytes)
        elapsed_time = end_time - start_time
        print(f'bandwidth sent: {send_data.nbytes/(end_time-start_time)/1024/1024} MB/s in {elapsed_time} seconds')

        # 接收数据
        start_time = time.time()
        recv_data = cp.zeros(test_send_size, dtype=cp.float32)
        comm.recv(recv_data.data.ptr, recv_data.size, nccl.NCCL_FLOAT, 1, cp.cuda.Stream.null.ptr)
        # cp.cuda.Stream.null.synchronize()
        end_time = time.time()
        print("Server received (iteration {i}):", recv_data, "Size:", recv_data.nbytes)
        elapsed_time = end_time - start_time
        print(f'bandwidth received: {recv_data.nbytes/(end_time-start_time)/1024/1024} MB/s in {elapsed_time} seconds')
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
