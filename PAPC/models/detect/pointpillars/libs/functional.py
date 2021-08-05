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

def mask_select(x, mask):
    mask_np = mask.numpy()
    mask_index = []
    for i in range(len(mask_np)):
        if mask_np[i] == True:
            mask_index.append(i)
        else:
            continue

    mask_index = paddle.to_tensor(mask_index)
    x = paddle.index_select(x, mask_index)

    return x

def select_change(x, y, index):
    x = x.numpy()
    x[:, index.numpy()] = y.numpy()

    return paddle.to_tensor(x)