"""
Modified from [minGPT](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py)
"""

import torch
import torch.nn as nn

class CasualSelfAttention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
