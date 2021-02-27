import paddle
import paddle.nn as nn

class VoxNet(nn.Layer):
    def __init__(self, name_scope='VoxNet_', num_classes=10):
        super(VoxNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv3D(1, 32, 5, 2),
            nn.BatchNorm(32),
            nn.LeakyReLU(),
            nn.Conv3D(32, 32, 3, 1),
            nn.MaxPool3D(2, 2, 0)
        )
        self.head = nn.Sequential(
            nn.Linear(32*6*6*6, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, inputs):
        x = paddle.to_tensor(inputs)
        x = self.backbone(x)
        x = paddle.reshape(x, (-1, 32*6*6*6))
        x = self.head(x)

        return x

if __name__ == '__main__':
    model = VoxNet()
    paddle.summary(model, (64, 1, 32, 32, 32))