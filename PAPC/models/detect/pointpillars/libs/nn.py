import paddle
import paddle.nn as nn

class Empty(nn.Layer):
    def __init__(self, *inputs, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *inputs, **kwargs):
        if len(inputs) == 1:
            return inputs[0]
        elif len(inputs) == 0:
            return None
        return inputs