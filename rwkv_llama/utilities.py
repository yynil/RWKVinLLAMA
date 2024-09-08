from typing import Optional
from transformers.cache_utils import Cache, DynamicCache
import torch
######state
class TimeMixState:
    def __init__(self, shift_state: torch.Tensor, wkv_state: torch.Tensor):
        self.shift_state = shift_state
        self.wkv_state = wkv_state


class ChannelMixState:
    def __init__(self, shift_state: torch.Tensor):
        self.shift_state = shift_state


class BlockState:
    def __init__(self, time_mix_state: TimeMixState,
                 channel_mix_state: ChannelMixState):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state

class BlockStateList:

    def __init__(self, shift_states, wkv_states):
        self.wkv_states = wkv_states
        self.shift_states = shift_states

    @staticmethod
    def create(N, B, C, H, device, dtype):
        result = BlockStateList.empty(N, B, C, H, device, dtype)
        result.wkv_states[:] = 0
        result.wkv_states[:] = 0
        result.shift_states[:] = 0
        return result

    @staticmethod
    def empty(N, B, C, H, device, dtype):
        wkv_states = torch.empty((N, B, H, C//H, C//H),
                                 device=device,
                                 dtype=torch.bfloat16)
        shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    def __getitem__(self, layer: int):
        return BlockState(
            TimeMixState(self.shift_states[layer, 0], self.wkv_states[layer]),
            ChannelMixState(self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state.shift_state
        self.wkv_states[layer] = state.time_mix_state.wkv_state
        self.shift_states[layer, 1] = state.channel_mix_state.shift_state

class HybridCache(DynamicCache):
    def update(self, 
               key_states, 
               value_states, 
               layer_idx, 
               cache_kwargs):
        # print("来自 HybridCache: update 方法被调用")
        return super().update(key_states, value_states, layer_idx, cache_kwargs)
    
    def get_seq_length(self, layer_idx: Optional[int] = 0):
        # print("来自 HybridCache: get_seq_length 方法被调用")
        return super().get_seq_length(layer_idx)
    
    def get_max_length(self):
        # print("来自 HybridCache: get_max_length 方法被调用")
        return super().get_max_length()
    
    def reorder_cache(self, beam_idx):
        # print("来自 HybridCache: reorder_cache 方法被调用")
        return super().reorder_cache(beam_idx)
    
    def __getitem__(self, item):
        # print("来自 HybridCache: __getitem__ 方法被调用")
        return super().__getitem__(item)