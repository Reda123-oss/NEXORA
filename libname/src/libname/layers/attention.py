import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional
import math
from .positional import rope   # must be: rope(x, start_pos=0), x of shape (B, T, H, D)


def repeat_kv(x , n_rep):
    B , n_kv_heads , T , head_dim = x.shape

    if n_rep == 1:
        return x

    x = x[ : , : , None , : , :]
    x = x.expand(B , n_kv_heads , n_rep , T , head_dim)
    x = x.reshape(B , n_kv_heads*n_rep , T , head_dim)
    return x


def create_attention_mask(T : int , device : torch.device , window_size : Optional[int] = None):
    positions = torch.arange(T , device = device)
    distance = (positions[: , None]-positions[None,:])
    mask = distance >= 0                                   # causal
    if window_size is not None:
        mask = mask & (distance < window_size)             # + sliding window
    return mask


class Attention(nn.Module):
    def __init__(self , config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.bias = config.bias

        self.n_kv_heads = getattr(config , "n_kv_heads" , None)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads                 # FIX: plain multi-head attention

        self.use_rope = getattr(config , "rope_theta" , None)
        self.window_size = getattr(config , "sliding_window" , None)
        self.max_seq_len = getattr(config , "max_seq_len" , 2048)   # size of the full KV cache

        assert self.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads

        self.Wq = nn.Linear(self.d_model , self.n_heads*self.head_dim , bias = False)
        self.Wk = nn.Linear(self.d_model , self.n_kv_heads*self.head_dim , bias = False)
        self.Wv = nn.Linear(self.d_model , self.n_kv_heads*self.head_dim , bias = False)
        self.Wo = nn.Linear(self.n_heads*self.head_dim , self.d_model , bias = self.bias)

        self.cache_k = None
        self.cache_V = None

    def forward(self , x , start_pos = None , use_cache = None):
        B , T , C = x.shape

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Q = Q.view(B , T , self.n_heads , self.head_dim)
        K = K.view(B , T , self.n_kv_heads , self.head_dim)
        V = V.view(B , T , self.n_kv_heads , self.head_dim)

        if not use_cache:                                   # FIX: use_cache=False also goes here
            if self.use_rope is not None:
                Q = rope(Q)
                K = rope(K)
            Q = Q.transpose(1,2)
            K = K.transpose(1,2)
            V = V.transpose(1,2)                            # FIX: V must be transposed too

            K = repeat_kv(K , self.n_rep)                   # FIX: repeat K and V (not Q and V)
            V = repeat_kv(V , self.n_rep)

            attn = torch.matmul(Q , K.transpose(-1,-2))
            attn = attn / math.sqrt(self.head_dim)          # FIX: scaling was missing
            # causal mask, plus the sliding window for Mistral / Mixtral
            mask = create_attention_mask(T , device = x.device , window_size = self.window_size)
            masked_attn = attn.masked_fill(~mask , float("-inf"))

            masked_attn = F.softmax(masked_attn , dim = -1)
            masked_attn = F.dropout(masked_attn , p = self.dropout , training = self.training)
            scores = torch.matmul(masked_attn , V)
            scores = scores.transpose(1,2).contiguous()
            scores = scores.view(B , T , self.n_heads*self.head_dim)
            out = self.Wo(scores)

        else:
            if start_pos is None:
                raise ValueError("start_pos is required when use_cache is True")

            if self.use_rope is not None:
                Q = rope(Q , start_pos)                     # FIX: positions start at start_pos
                K = rope(K , start_pos)
            Q = Q.transpose(1,2)
            K = K.transpose(1,2)
            V = V.transpose(1,2)                            # FIX

            if self.window_size is None:
                if self.cache_k is None or start_pos == 0:  # FIX: the cache was never allocated
                    if start_pos != 0:
                        raise ValueError("The first call must use start_pos = 0")
                    self.cache_k = torch.zeros(B , self.n_kv_heads , self.max_seq_len , self.head_dim , device = x.device , dtype = x.dtype)
                    self.cache_V = torch.zeros(B , self.n_kv_heads , self.max_seq_len , self.head_dim , device = x.device , dtype = x.dtype)
                if self.cache_k.shape[0] != B:
                    raise ValueError("The batch should match the one in the cache")
                if start_pos + T > self.max_seq_len:
                    raise ValueError("Sequence is longer than max_seq_len")

                self.cache_k[ : , : , start_pos : start_pos + T , :] = K
                self.cache_V[ : , : , start_pos : start_pos + T , :] = V

                cache_k = self.cache_k[ : , :,  : start_pos+T , :]
                cache_V = self.cache_V[ : , :, : start_pos +T , :]

                cache_k = repeat_kv(cache_k , self.n_rep)
                cache_V = repeat_kv(cache_V , self.n_rep)

                attn = torch.matmul(Q , cache_k.transpose(-1,-2))
                attn = attn/ math.sqrt(self.head_dim)

                # FIX: causal mask (needed when T > 1, i.e. during prefill)
                q_pos = torch.arange(start_pos , start_pos + T , device = x.device)
                k_pos = torch.arange(start_pos + T , device = x.device)
                mask = q_pos[ : , None] >= k_pos[None , : ]
                attn = attn.masked_fill(~mask , float("-inf"))

                masked_attn = F.softmax(attn , dim = -1)

                scores = torch.matmul(masked_attn , cache_V)   # FIX: use the cached values, not only V
                scores = scores.transpose(1,2).contiguous()
                scores = scores.view(B , T , self.n_heads*self.head_dim)   # FIX: not head_dim
                out = self.Wo(scores)
            else:
                W = self.window_size

                if self.cache_k is None or start_pos == 0:
                    if start_pos != 0:
                        raise ValueError("The first call must use start_pos = 0")
                    self.cache_k = torch.zeros(B , self.n_kv_heads , W , self.head_dim , device = x.device , dtype = x.dtype)
                    self.cache_V = torch.zeros(B , self.n_kv_heads , W , self.head_dim , device = x.device , dtype = x.dtype)
                if self.cache_k.shape[0] != B:
                    raise ValueError("The batch should match the one in the cache")

                old_start = max(0 , start_pos - W + 1)
                old_pos = torch.arange(old_start , start_pos , device = x.device)
                old_slots = old_pos % W
                old_k = torch.index_select(self.cache_k , 2 , old_slots)
                old_v = torch.index_select(self.cache_V , 2 , old_slots)
                all_K = torch.cat([old_k , K] , dim = 2)
                all_V = torch.cat([old_v , V] , dim = 2)

                all_K = repeat_kv(all_K , self.n_rep)        # FIX: grouped-query repeat
                all_V = repeat_kv(all_V , self.n_rep)

                attn = torch.matmul(Q , all_K.transpose(-2,-1))
                attn = attn / math.sqrt(self.head_dim)       # FIX: scaling
                new_positions = torch.arange(start_pos , start_pos + T , device = x.device)
                all_positions = torch.cat([old_pos , new_positions])
                distance = (new_positions[ : , None] - all_positions[None , :])
                mask = (distance >= 0) & (distance < W)
                masked_attn = attn.masked_fill(~mask , float("-inf"))
                masked_attn = F.softmax(masked_attn , dim = -1)
                scores = torch.matmul(masked_attn , all_V)
                scores = scores.transpose(1,2).contiguous()
                scores = scores.view(B , T , self.n_heads*self.head_dim)
                out = self.Wo(scores)

                with torch.no_grad():
                    for i in range(T):                       # later tokens overwrite older ones, so the last W tokens remain
                        slot = (start_pos + i) % W
                        self.cache_k[ : , : , slot , :] = K[ : , : , i , :]
                        self.cache_V[ : , : , slot , :] = V[ : , : , i , :]
        return out