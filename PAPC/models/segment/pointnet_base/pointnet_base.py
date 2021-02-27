import paddle
import paddle.nn as nn

class PointNet_Basic_Seg(nn.Layer):
    def __init__(self, num_classes=50, max_points=1024):
        super(PointNet_Basic_Seg, self).__init__()
        self.max_points = max_points
        self.pointnet_bacic = PointNet_Basic(max_points)
        self.seg_net = nn.Sequential(
            nn.Conv1D(max_points+64, 512, 1),
            nn.BatchNorm(512),
            nn.ReLU(),
            nn.Conv1D(512, 256, 1),
            nn.BatchNorm(256),
            nn.ReLU(),
            nn.Conv1D(256, 128, 1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv1D(128, 128, 1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv1D(128, num_classes, 1)
        )
    def forward(self, inputs):
        """
            Input:
                inputs: input points data, [B, 3, N]
            Return:
                x: predicts, [B, max_points, num_classes]
        """
        inputs = paddle.to_tensor(inputs[0])
        x1, x2 = self.pointnet_bacic(inputs)
        x2 = paddle.max(x2, axis=-1, keepdim=True)
        x2 = paddle.tile(x2, [1, 1, self.max_points])
        x = paddle.concat([x1, x2], axis=1)
        x = self.seg_net(x)
        x = paddle.squeeze(x, axis=-1)
        x = paddle.transpose(x, (0, 2, 1))

        return x

class PointNet_Basic(nn.Layer):
    def __init__(self, max_points=1024):
        super(PointNet_Basic, self).__init__()
        self.mlp_1 = nn.Sequential(
            nn.Conv1D(3, 64, 1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv1D(64, 64, 1),
            nn.BatchNorm(64),
            nn.ReLU(),
        )
        self.mlp_2 = nn.Sequential(
            nn.Conv1D(64, 64, 1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv1D(64, 128, 1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv1D(128, max_points, 1),
            nn.BatchNorm(max_points),
            nn.ReLU(),
        )
    def forward(self, inputs):
        """
            Input:
                inputs: input points data, [B, 3, N]
            Return:
                x1: points low feature, [B, C', N]
                x2: points high feature, [B, C'', N]
        """
        x1 = self.mlp_1(inputs)
        x2 = self.mlp_2(x1)

        return x1, x2

if __name__ == '__main__':
    model = PointNet_Basic_Seg()
    paddle.summary(model, (64, 3, 1024))