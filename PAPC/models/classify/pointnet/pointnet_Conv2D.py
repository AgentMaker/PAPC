import paddle
import paddle.nn as nn

class PointNet_Clas(paddle.nn.Layer):
    def __init__(self, num_classes=16, max_point=2048):
        super(PointNet_Clas, self).__init__()
        self.input_transform_net = nn.Sequential(
            nn.Conv2D(3, 64, (1, 1)),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv2D(64, 128, (1, 1)),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv2D(128, 1024, (1, 1)),
            nn.BatchNorm(1024),
            nn.ReLU(),
            nn.MaxPool2D((max_point, 1))
        )
        self.input_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9,
                weight_attr=paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Assign(paddle.zeros((256, 9)))),
                bias_attr=paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Assign(paddle.reshape(paddle.eye(3), [-1])))
            )
        )
        self.mlp_1 = nn.Sequential(
            nn.Conv2D(3, 64, (1, 1)),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv2D(64, 64,(1, 1)),
            nn.BatchNorm(64),
            nn.ReLU(),
        )
        self.feature_transform_net = nn.Sequential(
            nn.Conv2D(64, 64, (1, 1)),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv2D(64, 128, (1, 1)),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv2D(128, 1024, (1, 1)),
            nn.BatchNorm(1024),
            nn.ReLU(),

            nn.MaxPool2D((max_point, 1))
        )
        self.feature_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64*64)
        )
        self.mlp_2 = nn.Sequential(
            nn.Conv2D(64, 64, (1, 1)),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv2D(64, 128,(1, 1)),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv2D(128, 1024,(1, 1)),
            nn.BatchNorm(1024),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(axis=-1)
        )
    def forward(self, inputs):
        batchsize = inputs.shape[0]

        t_net = self.input_transform_net(inputs)
        t_net = paddle.squeeze(t_net, axis=[-2, -1])
        t_net = self.input_fc(t_net)
        t_net = paddle.reshape(t_net, [batchsize, 3, 3])

        x = paddle.squeeze(inputs, axis=-1)
        x = paddle.transpose(x, (0, 2, 1))
        x = paddle.matmul(x, t_net)
        x = paddle.transpose(x, (0, 2, 1))
        x = paddle.unsqueeze(x, axis=-1)
        x = self.mlp_1(x)

        t_net = self.feature_transform_net(x)
        t_net = paddle.squeeze(t_net, axis=[-2, -1])
        t_net = self.feature_fc(t_net)
        t_net = paddle.reshape(t_net, [batchsize, 64, 64])

        x = paddle.squeeze(x, axis=-1)
        x = paddle.transpose(x, (0, 2, 1))
        x = paddle.matmul(x, t_net)
        x = paddle.transpose(x, (0, 2, 1))
        x = paddle.unsqueeze(x, axis=-1)
        x = self.mlp_2(x)
        x = paddle.max(x, axis=2)
        x = paddle.squeeze(x, axis=-1)
        x = self.fc(x)

        return x