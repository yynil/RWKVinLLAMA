from typing import Optional
from transformers.cache_utils import Cache, DynamicCache

class HybridCache(DynamicCache):
    def update(self, key_states, value_states, layer_idx, cache_kwargs):
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
