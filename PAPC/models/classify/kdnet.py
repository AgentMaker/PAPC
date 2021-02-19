import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class KDNet(nn.Layer):
    def __init__(self, name_scope='KDNet_', num_classes=10):
        super(KDNet, self).__init__()
        self.conv1 = nn.Conv1D(3, 32*3, 1, 1)
        self.conv2 = nn.Conv1D(32, 64*3, 1, 1)
        self.conv3 = nn.Conv1D(64, 64*3, 1, 1)
        self.conv4 = nn.Conv1D(64, 128*3, 1, 1)
        self.conv5 = nn.Conv1D(128, 128*3, 1, 1)
        self.conv6 = nn.Conv1D(128, 256*3, 1, 1)
        self.conv7 = nn.Conv1D(256, 256*3, 1, 1)
        self.conv8 = nn.Conv1D(256, 512*3, 1, 1)
        self.conv9 = nn.Conv1D(512, 512*3, 1, 1)
        self.conv10 = nn.Conv1D(512, 128*3, 1, 1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, split_dims_v):
        def kdconv(x, dim, featdim, select, conv):
            x = F.relu(conv(x))
            x = paddle.reshape(x, (-1, featdim, 3, dim))
            x = paddle.reshape(x, (-1, featdim, 3 * dim))
            select = paddle.to_tensor(select) + (paddle.arange(0, dim) * 3)
            x = paddle.index_select(x, axis=2, index=select)
            x = paddle.reshape(x, (-1, featdim, int(dim/2), 2))
            x = paddle.max(x, axis=-1)

            return x

        x = kdconv(x, 1024, 32, split_dims_v[0], self.conv1)
        x = kdconv(x, 512, 64, split_dims_v[1], self.conv2)
        x = kdconv(x, 256, 64, split_dims_v[2], self.conv3)
        x = kdconv(x, 128, 128, split_dims_v[3], self.conv4)
        x = kdconv(x, 64, 128, split_dims_v[4], self.conv5)
        x = kdconv(x, 32, 256, split_dims_v[5], self.conv6)
        x = kdconv(x, 16, 256, split_dims_v[6], self.conv7)
        x = kdconv(x, 8, 512, split_dims_v[7], self.conv8)
        x = kdconv(x, 4, 512, split_dims_v[8], self.conv9)
        x = kdconv(x, 2, 128, split_dims_v[9], self.conv10)
        x = paddle.reshape(x, (-1, 128))
        x = self.fc(x)

        return x