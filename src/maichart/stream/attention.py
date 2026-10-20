"""Incremental attention using only the cross projections actually consumed."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from maichart.generation.event_model import _attention


def cached_decoder_layer(layer, hidden, memory, past_key, past_value,
                         past_memory_key, past_memory_value, additive_mask):
    """Same parameter/cache layout as the shared helper, without unused Q/K/V.

    A cross query needs Q only; immutable cross memory needs K/V only. The
    legacy helper projects all three in each case and throws the others away.
    """
    if hidden.size(1) != 1:
        raise ValueError('Cached stream attention requires one token')
    source=layer.norm1(hidden) if layer.norm_first else hidden
    query,key,value=F.linear(source,layer.self_attn.in_proj_weight,layer.self_attn.in_proj_bias).chunk(3,-1)
    keys=key if past_key is None else torch.cat((past_key,key),1)
    values=value if past_value is None else torch.cat((past_value,value),1)
    hidden=hidden+layer.dropout1(_attention(layer.self_attn,query,keys,values,None))
    if not layer.norm_first:hidden=layer.norm1(hidden)
    module=layer.multihead_attn
    width=module.embed_dim
    source=layer.norm2(hidden) if layer.norm_first else hidden
    bias=module.in_proj_bias
    query=F.linear(source,module.in_proj_weight[:width],None if bias is None else bias[:width])
    if past_memory_key is None or past_memory_value is None:
        memory_key,memory_value=F.linear(memory,module.in_proj_weight[width:],
            None if bias is None else bias[width:]).chunk(2,-1)
    else:memory_key,memory_value=past_memory_key,past_memory_value
    hidden=hidden+layer.dropout2(_attention(module,query,memory_key,memory_value,additive_mask))
    if not layer.norm_first:hidden=layer.norm2(hidden)
    source=layer.norm3(hidden) if layer.norm_first else hidden
    hidden=hidden+layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(source)))))
    if not layer.norm_first:hidden=layer.norm3(hidden)
    return hidden,(keys,values,memory_key,memory_value)
