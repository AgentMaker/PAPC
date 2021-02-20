import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class VFE_Clas(nn.Layer):
    def __init__(self, num_classes=16, max_points=1024):
        super(VFE_Clas, self).__init__()
        self.vfe = VFE(max_points=max_points)
        self.fc = self.fc = nn.Sequential(
            nn.Linear(max_points, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(256, num_classes)
        )
    def forward(self, inputs):
        """
            Input:
                inputs: input points data, [B, 3, N]
            Return:
                x: predicts, [B, num_classes]
        """
        x = paddle.to_tensor(inputs)
        x = self.vfe(x)
        x = self.fc(x)

        return x

class ConvBNReLU(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        super().__init__()
        self._conv = nn.Conv1D(in_channels, out_channels, kernel_size, stride, padding=padding, **kwargs)
        self._batch_norm = nn.BatchNorm(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = F.relu(x)
        return x

class PointNet_Basic(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(PointNet_Basic, self).__init__()
        self.mlp_1 = nn.Sequential(
            ConvBNReLU(in_channels, 64, 1),
            ConvBNReLU(64, 64, 1)
        )
        self.mlp_2 = nn.Sequential(
            ConvBNReLU(64, 64, 1),
            ConvBNReLU(64, 128, 1),
            ConvBNReLU(128, out_channels, 1)
        )
    def forward(self, inputs):
        """
            Input:
                inputs: input points data, [B, in_channels, N]
            Return:
                x: points feature, [B, out_channels, N]
        """
        x = self.mlp_1(inputs)
        x = self.mlp_2(x)

        return x

class VFE(nn.Layer):
    def __init__(self, feature_channels=256, max_points=1024):
        super(VFE, self).__init__()
        self.max_points = max_points
        self.pointnet_1 = PointNet_Basic(3, feature_channels)
        self.pointnet_2 = PointNet_Basic(feature_channels*2, max_points)
    def forward(self, inputs):
        """
            Input:
                inputs: input points data, [B, 3, N]
            Return:
                x: points feature, [B, C', N]
        """
        x1 = self.pointnet_1(inputs)
        x2 = paddle.max(x1, axis=-1, keepdim=True)
        x2 = paddle.tile(x2, [1, 1, self.max_points])
        x = paddle.concat([x1, x2], axis=1)
        x = self.pointnet_2(x)
        x = paddle.max(x, axis=-1)

        return x

if __name__ == '__main__':
    model = VFE_Clas()
    paddle.summary(model, (64, 3, 1024))