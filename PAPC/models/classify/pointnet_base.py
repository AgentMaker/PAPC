import paddle
import paddle.nn as nn

class PointNet_Basic_Clas(nn.Layer):
    def __init__(self, num_classes=10, max_points=1024):
        super(PointNet_Basic_Clas, self).__init__()
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
        self.fc = self.fc = nn.Sequential(
            nn.Linear(1024, 512),
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
        x = self.mlp_1(inputs)
        x = self.mlp_2(x)
        x = paddle.max(x, axis=2)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    model = PointNet_Basic_Clas()
    paddle.summary(model, (64, 3, 1024))