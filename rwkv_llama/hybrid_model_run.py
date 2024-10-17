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
from einops import rearrange
from fla.ops.rwkv6 import chunk_rwkv6,fused_recurrent_rwkv6
def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
    dtype = r.dtype
    r = rearrange(r, 'b l (h d) -> b h l d', h = H)
    k = rearrange(k, 'b l (h d) -> b h l d', h = H)
    v = rearrange(v, 'b l (h d) -> b h l d', h = H)
    w = rearrange(-torch.exp(w), 'b l (h d) -> b h l d', h = H)
    o, state = chunk_rwkv6(r, k, v, w, u=u, scale=1., initial_state=s, output_final_state=True,training=False)
    x = rearrange(o, 'b h l d -> b l (h d)')
    return x.to(dtype), state.to(dtype)
import torch
from utilities import TimeMixState, ChannelMixState, BlockState
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple
import logging
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
    
    
class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x060_infctx(args, layer_id)

        self.ffn = RWKV_CMix_x060_infctx(args, layer_id)
        
    def forward(self, x, last_state: BlockState = None, x_emb=None):
        args = self.args
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)
        if last_state is None:
            H =  args.dim_att // args.head_size_a
            device = x.device
            dtype = x.dtype
            wkv_states = torch.empty((B, H, C//H, C//H),
                                 device=device,
                                 dtype=dtype)
            shift_states = torch.empty((2,B,C),
                                 device=device,
                                 dtype=dtype)
            wkv_states[:] = 0
            shift_states[:] = 0
            time_state = TimeMixState(shift_states[0], wkv_states)
            # print(wkv_states)
            channel_state = ChannelMixState(shift_states[1])
            last_state = BlockState(time_state,channel_state)
        if self.layer_id == 0 and args.pre_ffn > 0:
            x = x + self.ffnPre(self.ln1(x))
        else:
            att_out, att_state = self.att(self.ln1(x), last_state.time_mix_state)
            x = x + att_out
        if args.is_llama_ffn:
            ffn_out = self.ffn(self.ln2(x))
            fnn_state = None
        else:
            ffn_out, fnn_state = self.ffn(self.ln2(x), last_state.channel_mix_state)
        x = x + ffn_out
        last_state.time_mix_state = att_state
        last_state.channel_mix_state = fnn_state
        return x, last_state
class RWKVDecoderLayer(nn.Module):
    def __init__(
        self,
        args,
        layer_idx: int
    ):
        super(RWKVDecoderLayer, self).__init__()
        self.block = Block(args,layer_idx)
        self.layer_idx = layer_idx
        self.args = args

    def forward(self, 
                hidden_states: torch.Tensor, 
                past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False, 
        output_attentions: Optional[bool] = False, 
        *args, 
        **kwargs):
        # Ensure hidden_states requires gradient
        _,T,_ = hidden_states.shape
        if past_key_value is not None:
            if len(past_key_value) <= self.layer_idx:
                last_state = None
            else:
                last_state = past_key_value[self.layer_idx][0]
        hidden_states,states= self.block(hidden_states,last_state)
        # hidden_states = self.block(hidden_states)
        # logging.info(f'forward in {self.layer_idx}')
        # so here is just to be compatible with Transformer

        # past_key_value = kwargs.get("past_key_value", None)

        if past_key_value is not None:
            keys = T
            values = states
            past_key_value.update(keys, values, self.layer_idx)
        outputs = (hidden_states,)
        if output_attentions :
            outputs += (None,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs

class HybridModel(nn.Module):
    def __init__(self,transformer_model,rwkv_args):
        super(HybridModel, self).__init__()
        attn_num_heads = transformer_model.config.num_attention_heads
        attn_num_key_value_heads = transformer_model.config.num_key_value_heads
        assert attn_num_heads % attn_num_key_value_heads == 0
        n_share = attn_num_heads // attn_num_key_value_heads
        def init_block_params(rwkv_args,layer_idx,llama_layer):
            if rwkv_args.is_rwkv_att_only:
                decoder = llama_layer
                att = RWKV_Tmix_x060_infctx_Wrapper(rwkv_args,layer_idx)
                # att.time_mixer.receptance.weight.data = llama_layer.self_attn.q_proj.weight.data
                # att.time_mixer.key.weight.data = llama_layer.self_attn.k_proj.weight.data.repeat(n_share, 1)
                # att.time_mixer.value.weight.data = llama_layer.self_attn.v_proj.weight.data.repeat(n_share, 1)
                # att.time_mixer.output.weight.data = llama_layer.self_attn.o_proj.weight.data
                llama_layer.self_attn = att
                return decoder
            else:
                decoder = RWKVDecoderLayer(rwkv_args,layer_idx)
                # decoder.block.att.receptance.weight.data = llama_layer.self_attn.q_proj.weight.data
                # decoder.block.att.key.weight.data = llama_layer.self_attn.k_proj.weight.data.repeat(n_share, 1)
                # decoder.block.att.value.weight.data = llama_layer.self_attn.v_proj.weight.data.repeat(n_share, 1)
                # decoder.block.att.output.weight.data = llama_layer.self_attn.o_proj.weight.data
                if rwkv_args.is_llama_ffn:
                    decoder.block.ffn = llama_layer.mlp
                return decoder
            # decoder = RWKVDecoderLayer(rwkv_args,layer_idx)
            # if rwkv_args.is_llama_ffn:
            #     decoder.block.ffn = llama_layer.mlp
            # decoder.block.att.receptance.weight.data = llama_layer.self_attn.q_proj.weight.data
            # decoder.block.att.key.weight.data = llama_layer.self_attn.k_proj.weight.data.repeat(n_share, 1)
            # decoder.block.att.value.weight.data = llama_layer.self_attn.v_proj.weight.data.repeat(n_share, 1)
            # decoder.block.att.output.weight.data = llama_layer.self_attn.o_proj.weight.data
            # decoder.block.ffn.key.weight.data = llama_layer.mlp.up_proj.weight.data
            # decoder.block.ffn.value.weight.data = llama_layer.mlp.down_proj.weight.data
            return decoder
        for layer_idx in range(transformer_model.config.num_hidden_layers):
            if layer_idx in rwkv_args.layers:
                rwkv_encoder = init_block_params(rwkv_args,layer_idx,transformer_model.model.layers[layer_idx])
                old_layer = transformer_model.model.layers[layer_idx]
                transformer_model.model.layers[layer_idx] = rwkv_encoder
                del old_layer
        self.model = transformer_model
        self.args = rwkv_args
    def forward(
        self,
        input_ids,
        inference_params=None,
        **kwargs,
    ):
        return self.model(input_ids, **kwargs)
    def load_ckpt(self, ckpt_file):
        print(f'loading ckpt from {ckpt_file}')
        info = self.load_state_dict(torch.load(ckpt_file,weights_only=True),strict=False)
        print(f'loaded ckpt info: {info}')
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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config_file = "configs/test_hybrid_full_logits_stage_2.yaml"
    import yaml
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    model_id = config['Llama']['model_id']
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    transformer_model = AutoModelForCausalLM.from_pretrained(model_id)
    print(transformer_model)
    args = create_rwkv_args(transformer_model.config, config)
    model = HybridModel(transformer_model,args)
    print(model)
    ckpt_file = '/data/rwkv/tmp/distill-c4-en-zh/pytorch_model.1400m.bin'
    model.load_ckpt(ckpt_file)