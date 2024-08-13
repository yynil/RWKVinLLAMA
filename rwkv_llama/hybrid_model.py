from src.model import RWKV_Tmix_x060, RWKV_CMix_x060,Block
import torch
import pytorch_lightning as pl
import torch.nn as nn
import deepspeed

class RWVVDecoderLayer(nn.Module):
    def __init__(
        self,
        args,
        layer_idx: int
    ):
        super(RWVVDecoderLayer, self).__init__()
        self.block = Block(args,layer_idx)
        self.layer_idx = layer_idx

    def forward(self, hidden_states: torch.Tensor, inference_params=None, *args, **kwargs):
        hidden_states = self.block(hidden_states)
        # print(f'forward in {self.layer_idx}')
        # so here is just to be compatible with Transformer

        past_key_value = kwargs.get("past_key_value", None)

        if past_key_value is not None:
            dummy_keys = torch.ones(
                1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
            )
            dummy_values = torch.ones(
                1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
            )
            # Update kv cache with dummy values
            past_key_value.update(dummy_keys, dummy_values, self.layer_idx)

        return (hidden_states, None, past_key_value)

class HybridModel(pl.LightningModule):
    def __init__(self,transformer_model,rwkv_args):
        super(HybridModel, self).__init__()
        self.emb = transformer_model.get_input_embeddings()
        attn_num_heads = transformer_model.config.num_attention_heads
        attn_num_key_value_heads = transformer_model.config.num_key_value_heads
        assert attn_num_heads % attn_num_key_value_heads == 0
        n_share = attn_num_heads // attn_num_key_value_heads
        def init_block_params(rwkv_args,layer_idx,llama_layer):
            decoder = RWVVDecoderLayer(rwkv_args,layer_idx)
            decoder.block.att.receptance.weight.data = llama_layer.self_attn.q_proj.weight.data
            decoder.block.att.key.weight.data = llama_layer.self_attn.k_proj.weight.data.repeat(n_share, 1)
            decoder.block.att.value.weight.data = llama_layer.self_attn.v_proj.weight.data.repeat(n_share, 1)
            decoder.block.att.output.weight.data = llama_layer.self_attn.o_proj.weight.data
            decoder.block.ffn.key.weight.data = llama_layer.mlp.up_proj.weight.data
            decoder.block.ffn.value.weight.data = llama_layer.mlp.down_proj.weight.data
            return decoder
        for layer_idx in range(transformer_model.config.num_hidden_layers):
            if layer_idx in rwkv_args.layers:
                rwkv_encoder = init_block_params(rwkv_args,layer_idx,transformer_model.model.layers[layer_idx])
                transformer_model.model.layers[layer_idx] = rwkv_encoder
        self.model = transformer_model
        self.rwkv_args = rwkv_args
    
    def forward(
        self,
        input_ids,
        inference_params=None,
        **kwargs,
    ):
        return self.model(input_ids, **kwargs)