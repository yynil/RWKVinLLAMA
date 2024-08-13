import cupy as cp
import numpy as np
from cupy.cuda import nccl

def client_process(nccl_id):
    # 设置 GPU 设备
    cp.cuda.Device(0).use()

    # 初始化 NCCL 通信器
    comm = nccl.NcclCommunicator(2, nccl_id, 1)

    # Client 逻辑
    for _ in range(5):  # 执行 5 次数据交换
        # 接收数据
        recv_data = cp.zeros(3, dtype=cp.float32)
        comm.recv(recv_data, 0)
        print("Client received:", recv_data)

        # 准备发送数据
        send_data = cp.array([4.0, 5.0, 6.0])
        comm.send(send_data, 0)
        print("Client sent:", send_data)

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