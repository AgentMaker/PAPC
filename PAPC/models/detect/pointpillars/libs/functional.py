import paddle
import math

def atan2(y, x):
    result = 0
    if x > 0:
        result = paddle.atan(y/x)
    elif x < 0 and y >= 0:
        result = paddle.atan(y/x) + math.pi
    elif x < 0 and y < 0:
        result = paddle.atan(y/x) - math.pi
    elif x == 0 and y > 0:
        result = paddle.to_tensor(math.pi)
    elif x == 0 and y < 0:
        result = paddle.to_tensor(-math.pi)
    else:
        pass

    return result