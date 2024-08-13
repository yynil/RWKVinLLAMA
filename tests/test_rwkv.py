import torch
import argparse
def setup_env():
    import os
    parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    rwkv_path = os.path.join(parent_path, 'rwkv')
    import sys
    sys.path.append(rwkv_path)
    print(f'add path: {rwkv_path} to sys.path')
    os.environ['RWKV_JIT_ON'] = '0'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ['RWKV_FLOAT_MODE'] = 'bf16'
    os.environ['RWKV_HEAD_SIZE_A'] = '64'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ["RWKV_MY_TESTING"]='x060'
    os.environ['RWKV_CTXLEN'] = '4096'
    os.environ['WKV'] = ''
    os.environ["RWKV_TRAIN_TYPE"] = ''
setup_env()
'''
parser.add_argument('--my_pos_emb', type=int, default=0, help='default 0, which is the position embedding in the model')
    parser.add_argument("--head_size_a", default=64, type=int) # can try larger values for larger models
    parser.add_argument('--head_size_divisor', type=int, default=8, help='head size divisor in the model')
    parser.add_argument('--ctx_len', type=int, default=4096, help='context length in the model')
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_qk", default=0, type=int)  # my headQK trick
    parser.add_argument("--tiny_att_dim", default=0, type=int)  # tiny attention dim
    parser.add_argument("--tiny_att_layer", default=-999, type=int)  # tiny attention @ which layer
    parser.add_argument("--vocab_size", default=65536, type=int)
'''
from argparse import Namespace
args = Namespace()
args.my_pos_emb = 0
args.head_size_a = 64
args.head_size_divisor = 8
args.ctx_len = 4096
args.n_layer = 6
args.n_embd = 512
args.dim_att = 512
args.dim_ffn = 0
args.pre_ffn = 0
args.head_qk = 0
args.tiny_att_dim = 0
args.tiny_att_layer = -999
args.vocab_size = 65536
args.dropout = 0
from src.model import RWKV
model = RWKV(args)
print(model)

