import sys
import os
def setup_env():
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(parent_dir)
    # rwkv_path = os.path.join(parent_dir, 'rwkv')
    # sys.path.append(rwkv_path)
    rwkv_llama_path = os.path.join(parent_dir, 'rwkv_llama')
    sys.path.append(rwkv_llama_path)
    # print(f'add path: {rwkv_path} to sys.path')
    print(f'add path: {rwkv_llama_path} to sys.path')
    os.environ['CUDA_HOME'] = '/usr/local/cuda-12.1'
    os.environ['RWKV_JIT_ON'] = '0'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ['RWKV_FLOAT_MODE'] = 'bf16'
    os.environ['RWKV_HEAD_SIZE_A'] = '64'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ["RWKV_MY_TESTING"]='x060'
    os.environ['RWKV_CTXLEN'] = '4096'
    os.environ['WKV'] = 'fla'
    os.environ["RWKV_TRAIN_TYPE"] = ''
setup_env()
import torch
if os.environ.get('WKV') == 'fla':
    from einops import rearrange
    from rwkvfla.ops.rwkv6 import chunk_rwkv6,fused_recurrent_rwkv6

    def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
        getattr(torch, r.device.type).set_device(r.device.index)
        r = rearrange(r, 'b l (h d) -> b h l d', h = H)
        k = rearrange(k, 'b l (h d) -> b h l d', h = H)
        v = rearrange(v, 'b l (h d) -> b h l d', h = H)
        w = rearrange(-torch.exp(w), 'b l (h d) -> b h l d', h = H)
        o, state = chunk_rwkv6(r, k, v, w, u=u, scale=1., initial_state=s, output_final_state=True,training=False)
        x = rearrange(o, 'b h l d -> b l (h d)')
        return x, state
else:
    from torch.utils.cpp_extension import load
    HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    src_path = os.path.join(parent_dir,'rwkv')
    wkv6state_cuda = load(name="wkv6infctx", sources=[f"{src_path}/cuda/wkv6infctx_op.cpp", f"{src_path}/cuda/wkv6infctx_cuda.cu"],
                            verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
                
    class WKV_6STATE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, H, r, k, v, w, u, s):
            with torch.no_grad():
                assert r.dtype == torch.bfloat16
                assert k.dtype == torch.bfloat16
                assert v.dtype == torch.bfloat16
                assert w.dtype == torch.bfloat16
                assert u.dtype == torch.bfloat16
                assert s.dtype == torch.bfloat16
                assert HEAD_SIZE == C // H
                ctx.B = B
                ctx.T = T
                ctx.C = C
                ctx.H = H
                assert r.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert w.is_contiguous()
                assert u.is_contiguous()
                assert s.is_contiguous()
                ctx.save_for_backward(r, k, v, w, u, s)
                y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                wkv6state_cuda.forward(B, T, C, H, r, k, v, w, u, s, y)
                return y

        @staticmethod
        def backward(ctx, gy):
            with torch.no_grad():
                assert gy.dtype == torch.bfloat16
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                assert gy.is_contiguous()
                r, k, v, w, u, s = ctx.saved_tensors
                gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gs = torch.empty((B, H, C//H, C//H), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                wkv6state_cuda.backward(B, T, C, H, r, k, v, w, u, s, gy, gr, gk, gv, gw, gu, gs)
                gu = torch.sum(gu, 0).view(H, C//H)
                gs = torch.sum(gs, 0).view(H, C//H, C//H)
                return (None, None, None, None, gr, gk, gv, gw, gu, gs)

    def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
        x = WKV_6STATE.apply(B, T, C, H, r, k, v, w, u, s)
        return x, s
import torch
from utilities import TimeMixState, ChannelMixState, BlockState
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple
import logging
from transformers import AutoConfig,AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights
from transformers.cache_utils import Cache,DynamicCache
# from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
import os
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
class RWKV_Tmix_x060_infctx(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32 # generate TIME_MIX for w,k,v,r,g
            if args.n_embd==4096:
                D_MIX_LORA = D_MIX_LORA*2
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

            D_DECAY_LORA = 64
            if args.n_embd==4096:
                D_DECAY_LORA = D_DECAY_LORA*2
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))
            #self.time_state = nn.Parameter(torch.zeros(self.n_head, self.head_size, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

    def jit_func(self, x, shift_state):
        B, T, C = x.size()
        if shift_state is not None:
            xx = torch.concat((shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        else:
            xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w, x[:, -1]

    def jit_func_2(self, x, g, timemixstate:TimeMixState):
        B, T, C = x.size()
        x = x.view(B * T, C)
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x, timemixstate

    def forward(self, x, last_state: TimeMixState):
        B, T, C = x.size()
        H = self.n_head
        shift_state = last_state.shift_state
        r, k, v, g, w, lx = self.jit_func(x, shift_state)
        ######
        wkv_state = last_state.wkv_state
        x, wkv_state = RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u=self.time_faaaa, s=wkv_state)
        return self.jit_func_2(x, g, TimeMixState(lx, wkv_state))
    
class RWKV_Tmix_x060_infctx_Wrapper(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_idx = layer_id
        self.time_mixer = RWKV_Tmix_x060_infctx(args, layer_id)

    def forward(self, 
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
            **kwargs):
        x = hidden_states
        args = self.args
        B, T, C = x.size()
        if past_key_value is not None:
            if len(past_key_value) <= self.layer_idx:
                last_state = None
            else:
                last_state = past_key_value[self.layer_idx][0]
        if last_state is None:
            H =  args.dim_att // args.head_size_a
            device = x.device
            dtype = x.dtype
            wkv_states = torch.empty((B, H, C//H, C//H),
                                 device=device,
                                 dtype=dtype)
            token_shift = torch.empty((B,C),
                                 device=device,
                                 dtype=dtype)
            wkv_states[:] = 0
            token_shift[:] = 0
            time_state = TimeMixState(token_shift, wkv_states)
            # print(wkv_states)
            channel_state = None
            last_state = BlockState(time_state,channel_state)
        x,states= self.time_mixer(x,last_state.time_mix_state)
        last_state.time_mix_state = states
        if past_key_value is not None:
            keys = T
            values = last_state
            past_key_value.update(keys, values, self.layer_idx)
        return x,None,past_key_value

class RWKV_CMix_x060_infctx(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x, last_state: ChannelMixState):
        if last_state.shift_state is not None:
            xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        else:
            xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv, ChannelMixState(x[:, -1])
    


class HybridModel(nn.Module):
    
    def __init__(self,rwkv_args,transformer_config):
        super(HybridModel, self).__init__()
        self.args = rwkv_args
        print(f'rwkv_args: {rwkv_args}')
        print(f'transformer_config: {transformer_config}')
        if transformer_config.tie_word_embeddings :
            transformer_config.tie_word_embeddings = False
        with no_init_weights():
            self.model = AutoModelForCausalLM.from_config(transformer_config)
        #Replace the self attention to TimeMixer
        for layer_idx in range(transformer_config.num_hidden_layers):
            llama_layer = self.model.model.layers[layer_idx]
            if layer_idx in rwkv_args.layers:
                att = RWKV_Tmix_x060_infctx_Wrapper(rwkv_args,layer_idx)
                old_attn = llama_layer.self_attn
                llama_layer.self_attn = att
                del old_attn
                print(f'layer {layer_idx} is replaced by RWKV TimeMixer_x060')
        import gc
        gc.collect()
    def load_checkpoint(self, path):
        all_keys = set(self.state_dict().keys())
        incompatible_keys = set()
        #if the path is the file, load it directly
        #if the path is the directory, load the sharded files in the directory with suffix .pt
        if os.path.isdir(path):
            files = os.listdir(path)
            files = [os.path.join(path, f) for f in files if f.endswith('.pt')]
        else:
            files = [path]
        for file in files:
            checkpoint = torch.load(file, map_location='cpu')
            self.load_state_dict(checkpoint, strict=False)
            print(f'load model from {file}')
            ckpt_keys = checkpoint.keys()
            #subtract the keys in the checkpoint from the all_keys
            #if the ckpt_key exists in the all_keys, remove it
            for ckpt_key in ckpt_keys:
                if ckpt_key in all_keys:
                    all_keys.remove(ckpt_key)
                else:
                    incompatible_keys.add(ckpt_key)
            del checkpoint
            import gc
            gc.collect()
        print(f'Finish loading model from {path}')
        print(f'Incompatible keys: {incompatible_keys} missing keys: {all_keys}')
        
    def warmup_all_group_norms(self):
        """预热所有 RWKV 层中的 GroupNorm"""
        print("Starting GroupNorm warmup for all layers")
        
        with torch.no_grad():
            num_gpus = torch.cuda.device_count()
            
            for layer_idx in self.args.layers:
                print(f"Warming up layer {layer_idx}")
                layer = self.model.model.layers[layer_idx].self_attn.time_mixer
                
                device = layer.time_maa_x.device
                print(f"Warming up on {device}")
                
                torch.cuda.synchronize(device)
                # 创建测试输入
                dummy_batch = torch.ones(2, 1, self.args.dim_att,
                                      dtype=torch.bfloat16,
                                      device=device)
                dummy_state = TimeMixState(
                    shift_state=torch.zeros(2, self.args.dim_att,
                                          dtype=torch.bfloat16,
                                          device=device),
                    wkv_state=torch.zeros(2, layer.n_head,
                                        self.args.dim_att//layer.n_head,
                                        self.args.dim_att//layer.n_head,
                                        dtype=torch.bfloat16,
                                        device=device)
                )
                
                try:
                    # 运行一次完整的前向传播
                    _, _ = layer(dummy_batch, dummy_state)
                    print(f"Successfully warmed up layer {layer_idx} on {device}")
                except Exception as e:
                    print(f"Warning: Failed to warm up layer {layer_idx} on {device}: {e}")
                
                # 清理内存
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
        
        print("Finished warming up all layers")
    
    def forward(
        self,
        input_ids,
        inference_params=None,
        **kwargs,
    ):
        return self.model(input_ids, **kwargs)
def create_rwkv_args(transformer_config, config):
    from argparse import Namespace
    args = Namespace()
    args.layers = config['RWKV']['layers']
    args.my_pos_emb = 0
    args.head_size_a = 64
    args.head_size_divisor = 8
    args.ctx_len = 4096
    args.n_layer = transformer_config.num_hidden_layers
    args.n_embd = transformer_config.hidden_size
    args.dim_att = transformer_config.hidden_size
    args.dim_ffn = transformer_config.intermediate_size
    args.pre_ffn = 0
    args.head_qk = 0
    args.tiny_att_dim = 0
    args.tiny_att_layer = -999
    args.vocab_size = transformer_config.vocab_size
    args.pad_id = transformer_config.pad_token_id
    args.is_llama_ffn = config.get('is_llama_ffn',False)
    args.is_rwkv_att_only = config.get('is_rwkv_att_only',False)
    return args
         
if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
    config_file = "configs/qwen_32b.yaml"
    import yaml
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    model_id = config['Llama']['model_id']
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    transformer_config = AutoConfig.from_pretrained(model_id)
    args = create_rwkv_args(transformer_config, config)
    model = HybridModel(args,transformer_config)
    print(model)
    ckpt_file = '/home/yueyulin/model/qwen_32b_distill_4k'
    model.load_checkpoint(ckpt_file)
    dtype = torch.bfloat16
    model = model.to(dtype=dtype)
    num_layers = model.model.config.num_hidden_layers
    num_gpus = 4
    device_map = {}
    average_layers = num_layers // num_gpus
    for i in range(num_layers):
        device_map[f'model.layers.{i}'] = i // average_layers
    device_map['model.embed_tokens'] = 'cpu'
    device_map['model.norm'] = 'cpu'
    device_map['model.rotary_emb'] = 'cpu'
    device_map['lm_head'] = 'cpu'
    from accelerate import dispatch_model
    model.model = dispatch_model(model.model, device_map=device_map,offload_buffers=True)
    print(model)
    input('press any key to continue')
    print('Model warmup start')
    model.warmup_all_group_norms()
    print('Model warmup end')
    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        max_new_tokens=256,
        stop_strings = ["<|im_end|>"],
        do_sample = True,
        use_cache = True,
        temperature = 0.3,
        top_k = 20,
        top_p = 0.5,
        min_p = 0.05,
        repetition_penalty = 1.1,
        no_repeat_ngram_size = 3,
    )
    message = input('please input message:')
    conversation = [{
        'role': 'user',
        'content': message
    }]
    while True:
        current_input_text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        index_of_start = current_input_text.find("<|im_start|>user")
        if index_of_start != -1:
            current_input_text = current_input_text[index_of_start:]
        print(current_input_text)
        input_ids = tokenizer(current_input_text, return_tensors="pt").to("cuda:0")
        input_length = input_ids.input_ids.shape[1]
        from rwkv_llama.utilities import HybridCache
        cache = HybridCache()
        with torch.no_grad():
            model_to_use = model.model
            print(model_to_use)
            output = model_to_use.generate(
                    input_ids=input_ids['input_ids'],
                    attention_mask=input_ids['attention_mask'],
                    past_key_values=cache,
                    generation_config=gen_config,
                    tokenizer = tokenizer,
                    use_cache = True
                )
        
        generated_text = tokenizer.decode(output[0,input_length:], skip_special_tokens=True)            
        print(generated_text)
        conversation.append({
            'role': 'assistant',
            'content': generated_text
        })
        message = input(':')
        conversation.append({
            'role': 'user',
            'content': message
        })
        if message == 'exit':
            break