import cupy as cp
import numpy as np
from cupy.cuda import nccl
def enable_p2p():
    num_gpus = cp.cuda.runtime.getDeviceCount()
    for i in range(num_gpus):
        for j in range(num_gpus):
            if i != j:
                can_access_peer = cp.cuda.runtime.deviceCanAccessPeer(i, j)
                if can_access_peer:
                    cp.cuda.runtime.deviceEnablePeerAccess(j)
                    print(f"Enabled P2P access between GPU {i} and GPU {j}")
                else:
                    print(f"Cannot enable P2P access between GPU {i} and GPU {j}")
enable_p2p()
def client_process(nccl_id):
    # 设置 GPU 设备
    cp.cuda.Device(1).use()

    # 初始化 NCCL 通信器
    comm = nccl.NcclCommunicator(2, nccl_id, 1)
    test_send_size = 2048*4096*20
    import time
    # Client 逻辑
    for i in range(5):  # 执行 5 次数据交换
        # 接收数据
        start_time = time.time()
        recv_data = cp.zeros(test_send_size, dtype=cp.float32)
        comm.recv(recv_data.data.ptr, recv_data.size, nccl.NCCL_FLOAT, 0, cp.cuda.Stream.null.ptr)
        # cp.cuda.Stream.null.synchronize()
        end_time = time.time()
        print(f"Client received (iteration {i}):", recv_data, "Size:", recv_data.nbytes)
        print(f"bandwidth received: {recv_data.nbytes/(end_time-start_time)/1024/1024} MB/s")

        # 准备发送数据
        start_time = time.time()
        send_data = cp.array([4.0]*test_send_size,dtype=cp.float32)
        comm.send(send_data.data.ptr, send_data.size, nccl.NCCL_FLOAT, 0, cp.cuda.Stream.null.ptr)
        # cp.cuda.Stream.null.synchronize()
        end_time = time.time()
        print(f"Client sent (iteration {i}):", send_data, "Size:", send_data.nbytes)
        print(f"bandwidth sent: {send_data.nbytes/(end_time-start_time)/1024/1024} MB/s")

    print("Client finished")

if __name__ == "__main__":
    # 读取 NCCL ID
    import sys
    nccl_id_file = sys.argv[1]
    with open(nccl_id_file, 'r') as f:
        import json
        nccl_id = json.load(f)['nccl_id']
        nccl_id = tuple(nccl_id)
    print("NCCL ID:", nccl_id)

    client_process(nccl_id)