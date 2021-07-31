import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fluid.core as core

import inspect
import numpy as np 

from libs.tools.checkpoint import *
from libs.tools.optim import *

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups, eps=1e-5, affine=True):
        super().__init__(
            num_groups=num_groups,
            num_channels=num_channels,
            epsilon=eps,
            weight_attr=True,
            bias_attr=True)

def one_hot(tensor, depth, dtype="float32"):
    tensor_onehot = F.one_hot(tensor.astype("int64"), depth).astype(dtype)

    return tensor_onehot

def get_paddings_indicator(actual_num,max_num,axis = 0):
    '''Create boolean mask by actually number of a padded tensor.'''
    actual_num = paddle.unsqueeze(actual_num, axis+1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis+1] = -1
    max_num = paddle.arange(max_num, dtype="int64").reshape(max_num_shape)

    paddings_indicator = actual_num.int() > max_num

    return paddings_indicator

def get_pos_to_kw_map(func):
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw

def get_kw_to_default_map(func):
    kw_to_default = {}
    fsig = inspect.signature(func)
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            if info.default is not info.empty:
                kw_to_default[name] = info.default
    return kw_to_default

def change_default_args(**kwargs):
    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class):
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)

        return DefaultArgLayer

    return layer_wrapper

def paddle_to_np_dtype(ttype):
    type_map = {
        core.VarDesc.VarType.FP16: np.dtype(np.float16),
        core.VarDesc.VarType.FP32: np.dtype(np.float32),
        core.VarDesc.VarType.FP64: np.dtype(np.int32),
        core.VarDesc.VarType.INT64: np.dtype(np.int64),
        core.VarDesc.VarType.UINT8: np.dtype(np.uint8),
    }
    return type_map[ttype]
