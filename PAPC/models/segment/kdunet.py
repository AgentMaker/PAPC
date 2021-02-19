import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class KDUNet(nn.Layer):
    def __init__(self, num_classes=50):
        super(KDUNet, self).__init__()
        self.downsample = Downsample()
        self.upsample = Upsample(num_classes)

    def forward(self, x, split_dims_v):
        x, shortcut = self.downsample(x, split_dims_v)
        x = self.upsample(x, shortcut)

        return x

class ConvBNReLU(nn.Layer):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding='same',
                **kwargs):
        super().__init__()

        self._conv = nn.Conv1D(
            in_channels, out_channels, kernel_size, stride, padding=padding, **kwargs)

        self._batch_norm = nn.BatchNorm(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = F.relu(x)
        return x

class Downsample(nn.Layer):
    def __init__(self):
        super(Downsample, self).__init__()
        self.convbnrelu1 = ConvBNReLU(3, 32 * 3, 1, 1)
        self.convbnrelu2 = ConvBNReLU(32, 64 * 3, 1, 1)
        self.convbnrelu3 = ConvBNReLU(64, 256 * 3, 1, 1)
        self.convbnrelu4 = ConvBNReLU(256, 512 * 3, 1, 1)
        self.convbnrelu5 = ConvBNReLU(512, 1024 * 3, 1, 1)

    def forward(self, x, split_dims_v):
        def kdconv(x, shortcut, dim, featdim, select, convbnrelu):
            shortcut.append(x)
            x = convbnrelu(x)
            x = paddle.reshape(x, (-1, featdim, 3, dim))
            x = paddle.reshape(x, (-1, featdim, 3 * dim))
            select = paddle.to_tensor(select) + (paddle.arange(0, dim) * 3)
            x = paddle.index_select(x, axis=2, index=select)
            x = paddle.reshape(x, (-1, featdim, int(dim / 2), 2))
            x = paddle.max(x, axis=-1)

            return x, shortcut

        shortcut = []

        x, shortcut = kdconv(x, shortcut, 1024, 32, split_dims_v[0], self.convbnrelu1)
        x, shortcut = kdconv(x, shortcut, 512, 64, split_dims_v[1], self.convbnrelu2)
        x, shortcut = kdconv(x, shortcut, 256, 256, split_dims_v[2], self.convbnrelu3)
        x, shortcut = kdconv(x, shortcut, 128, 512, split_dims_v[3], self.convbnrelu4)
        x, shortcut = kdconv(x, shortcut, 64, 1024, split_dims_v[4], self.convbnrelu5)

        return x, shortcut


class Upsample(nn.Layer):
    def __init__(self, num_classes=50):
        super(Upsample, self).__init__()
        self.deconv1 = nn.Conv1DTranspose(1024, 512, 2, 2)
        self.doubleconv1 = nn.Sequential(
            ConvBNReLU(1024, 512, 1, 1),
            ConvBNReLU(512, 512, 1, 1))
        self.deconv2 = nn.Conv1DTranspose(512, 512, 2, 2)
        self.doubleconv2 = nn.Sequential(
            ConvBNReLU(768, 512, 1, 1),
            ConvBNReLU(512, 512, 1, 1))
        self.deconv3 = nn.Conv1DTranspose(512, 256, 2, 2)
        self.doubleconv3 = nn.Sequential(
            ConvBNReLU(320, 256, 1, 1),
            ConvBNReLU(256, 256, 1, 1))
        self.deconv4 = nn.Conv1DTranspose(256, 256, 2, 2)
        self.doubleconv4 = nn.Sequential(
            ConvBNReLU(288, 128, 1, 1),
            ConvBNReLU(128, 128, 1, 1))
        self.deconv5 = nn.Conv1DTranspose(128, 128, 2, 2)
        self.doubleconv5 = nn.Sequential(
            ConvBNReLU(131, 128, 1, 1),
            nn.Conv1D(128, num_classes, 1, 1))

    def forward(self, x, shortcut):
        x = self.deconv1(x)
        x = paddle.concat([x, shortcut[-1]], axis=1)
        x = self.doubleconv1(x)
        x = self.deconv2(x)
        x = paddle.concat([x, shortcut[-2]], axis=1)
        x = self.doubleconv2(x)
        x = self.deconv3(x)
        x = paddle.concat([x, shortcut[-3]], axis=1)
        x = self.doubleconv3(x)
        x = self.deconv4(x)
        x = paddle.concat([x, shortcut[-4]], axis=1)
        x = self.doubleconv4(x)
        x = self.deconv5(x)
        x = paddle.concat([x, shortcut[-5]], axis=1)
        x = self.doubleconv5(x)

        return x