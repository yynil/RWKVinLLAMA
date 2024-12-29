import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load
HEAD_SIZE = 64
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'parent_dir: {parent_dir}')
AUTO_RESCALE = os.getenv('AUTO_RESCALE', '1')== '1'
is_wind_cuda = False
if 'is_wind_cuda' in os.environ:
    is_wind_cuda = os.environ['is_wind_cuda'] == '1'
if is_wind_cuda:    
    load(name="wind", sources=[f'{parent_dir}/rwkv_cuda_wind/wind_rwkv7.cu', f'{parent_dir}/rwkv_cuda_wind/wind_rwkv7.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=[f'-D_C_={HEAD_SIZE}',"-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"])

    class WindRWKV7(torch.autograd.Function):
        @staticmethod
        def forward(ctx,w,q,k,v,a,b):
            B,T,H,C = w.shape
            s0 = torch.zeros(B,H,C,C,dtype=w.dtype,device=w.device)
            assert T%16 == 0
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,a,b,s0])
            w,q,k,v,a,b,s0 = [i.contiguous() for i in [w,q,k,v,a,b,s0]]
            y = torch.empty_like(v)
            sT = torch.empty_like(s0)
            s = torch.zeros(B,H,T//16,C,C, dtype=w.dtype,device=w.device)
            torch.ops.wind.forward(w,q,k,v,a,b, s0,y,s,sT)
            ctx.save_for_backward(w,q,k,v,a,b,s)
            return y
        
        @staticmethod
        def backward(ctx,dy):
            w,q,k,v,a,b,s = ctx.saved_tensors
            B,T,H,C = w.shape
            dsT = torch.zeros(B,H,C,C,dtype=dy.dtype,device=dy.device)
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            dy,dsT = [i.contiguous() for i in [dy,dsT]]
            dw,dq,dk,dv,da,db,ds0 = [torch.empty_like(x) for x in [w,q,k,v,a,b,dsT]]
            torch.ops.wind.backward(w,q,k,v,a,b, dy,s,dsT, dw,dq,dk,dv,da,db,ds0)
            return dw,dq,dk,dv,da,db

    def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
        B,T,HC = q.shape
        q,w,k,v,a,b = [i.view(B,T,HC//HEAD_SIZE,HEAD_SIZE) for i in [q,w,k,v,a,b]]
        return WindRWKV7.apply(w,q,k,v,a,b).view(B,T,HC)
else:#use fast
    CHUNK_LEN = 16

    flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    load(name="wind_backstepping", sources=[f'{parent_dir}/cuda/wkv7_cuda.cu', f'{parent_dir}/cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w,q,k,v,z,b):
            B,T,H,C = w.shape 
            assert T%CHUNK_LEN == 0
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
            assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
            torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
            ctx.save_for_backward(w,q,k,v,z,b,s,sa)
            return y
        @staticmethod
        def backward(ctx, dy):
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w,q,k,v,z,b,s,sa = ctx.saved_tensors
            dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
            torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
            return dw,dq,dk,dv,dz,db

    def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
        B,T,HC = q.shape
        q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
        return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)
    
########################################################################################################
# RWKV TimeMix
########################################################################################################
def safe_check(x, name, layer_id):
    if torch.isnan(x).any():
        max_val = torch.max(torch.abs(x[~torch.isnan(x)])) if (~torch.isnan(x)).any() else float('nan')
        print(f"NaN detected in layer {layer_id}, {name}, max non-nan value: {max_val}")
        return True
    return False

def safe_matrix_mult(x, weight, layer_id, name):
    """Safe mamul"""
    result = x @ weight
    if safe_check(result, f"{name} matrix multiplication", layer_id):
        # in case NaN is detected, try to use a more conservative way
        # 1. Normalize x
        x_normalized = x / (torch.max(torch.abs(x)) + 1e-5)
        # 2. Multiply
        result = x_normalized @ weight
        # 3. Check result
        if safe_check(result, f"{name} safe multiplication", layer_id):
            # in case NaN is still detected, scale down the result
            x_scaled = x_normalized * 0.01  # scale down to 1%
            result = x_scaled @ weight
    return result
class RWKV_Tmix_x070(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.bf16_limit = 2**8  # bf16 的有效范围大约是 2^8
        self.rescale_threshold = self.bf16_limit * 0.7  # 设置阈值为极限的70%

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd
        self.first_nan_detected = False

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_DECAY_LORA = 64
            # D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = 64
            # D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = 32
            # D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            D_GATE_LORA = 128
            # D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(args.head_size_divisor**2)) # !!! notice eps value !!!

            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            # self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            # self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.output.weight.data.zero_()
            
    def check_needs_rescale(self, tensor):
        if not AUTO_RESCALE:
            return False
            
        max_abs = torch.max(torch.abs(tensor))
        return max_abs > self.rescale_threshold
    
    def debug_nan(self, tensor, name):
        if not self.first_nan_detected and torch.isnan(tensor).any():
            self.first_nan_detected = True
            valid_values = tensor[~torch.isnan(tensor)]
            stats = ""
            if len(valid_values) > 0:
                stats = f"\n    Non-NaN stats - Min: {torch.min(valid_values).item():.4f}, Max: {torch.max(valid_values).item():.4f}, Mean: {torch.mean(valid_values).item():.4f}"
            
            print(f"First NaN detected in layer {self.layer_id}, {name}:{stats}")
            return True
        return False

    
    
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        # Check if input tensor has NaN
        xx = self.time_shift(x) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            # # 1. Safe matrix multiplication
            # v1_result = safe_matrix_mult(xv, self.v1, self.layer_id, "v1")
            # v2_result = safe_matrix_mult(v1_result, self.v2, self.layer_id, "v2")
            
            # # 2. make sure the input is in a safe range
            # gate_input = self.v0 + v2_result
            
            # # 3. make sure the gate is in a safe range
            # max_abs_gate = torch.max(torch.abs(gate_input))
            # if max_abs_gate > 16:
            #     gate_input = gate_input * (16 / max_abs_gate)
                
            # gate = torch.sigmoid(gate_input)
            # safe_check(gate, "gate", self.layer_id)
            
            # # 4. calculate the difference and then add the residual
            # v_diff = v_first - v
            # safe_check(v_diff, "v_diff", self.layer_id)
            
            # # 5. make sure the diff is in a safe range
            # max_diff = torch.max(torch.abs(v_diff))
            # if max_diff > 1:
            #     v_diff = v_diff / max_diff
                
            # # 6. Update the final residual
            # v_new = v + v_diff * gate
            
            # # 7. Make the final range check
            # if safe_check(v_new, "v_new", self.layer_id):
            #     # if NaN is detected, try to use a more conservative way
            #     epsilon = 1e-6
            #     safe_gate = torch.clamp(gate, epsilon, 1-epsilon)
            #     v_new = v + v_diff * safe_gate * 0.1  
                
            # v = v_new
            
            ###Original implementation
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual

            # 8. Check if v needs rescale (the values are too large)
            # if self.check_needs_rescale(v):
            #     max_val = torch.max(torch.abs(v))
            #     scale = self.rescale_threshold / (max_val + 1e-5)
            #     v = v * scale
            #     print(f'Layer {self.layer_id} rescaled v by {scale:.4f}')
                # if v_first is not None:
                #     v_first = v_first * scale
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)
        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        # needs_rescale = self.check_needs_rescale(x)
        # if needs_rescale:
        #     x = x / 2
        #     print(f'Layer {self.layer_id} rescaled x by 0.5')
        #     # v_first = v_first / 2 if v_first is not None else None

        return x, v_first
    

    
########################################################################################################
# RWKV ChannelMix
########################################################################################################
class RWKV_CMix_x070(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        # self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        # self.value.weight.data.zero_()

    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)

########################################################################################################
# RWKV Block
########################################################################################################

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
    def forward(self, x,v_first):

        if self.layer_id == 0:
            x = self.ln0(x)
        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn
        x = x + self.ffn(self.ln2(x))

        
        return x,v_first