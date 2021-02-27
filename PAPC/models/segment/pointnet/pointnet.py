import paddle
import paddle.nn as nn

class PointNet_Seg(nn.Layer):
    def __init__(self, num_classes=50, max_point=2048):
        super(PointNet_Seg, self).__init__()
        self.max_point = max_point
        self.input_transform_net = nn.Sequential(
            nn.Conv1D(3, 64, 1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv1D(64, 128, 1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv1D(128, 1024, 1),
            nn.BatchNorm(1024),
            nn.ReLU(),
            nn.MaxPool1D(max_point)
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
            nn.Conv1D(3, 64, 1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv1D(64, 64, 1),
            nn.BatchNorm(64),
            nn.ReLU(),
        )
        self.feature_transform_net = nn.Sequential(
            nn.Conv1D(64, 64, 1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv1D(64, 128, 1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv1D(128, 1024, 1),
            nn.BatchNorm(1024),
            nn.ReLU(),

            nn.MaxPool1D(max_point)
        )
        self.feature_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64*64)
        )
        self.mlp_2 = nn.Sequential(
            nn.Conv1D(64, 64, 1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv1D(64, 128, 1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv1D(128, 1024, 1),
            nn.BatchNorm(1024),
            nn.ReLU(),
        )
        self.seg_net = nn.Sequential(
            nn.Conv1D(1024+64, 512, 1),
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
        inputs = paddle.to_tensor(inputs[0])
        batchsize = inputs.shape[0]

        t_net = self.input_transform_net(inputs)
        t_net = paddle.squeeze(t_net, axis=-1)
        t_net = self.input_fc(t_net)
        t_net = paddle.reshape(t_net, [batchsize, 3, 3])

        x = paddle.transpose(inputs, (0, 2, 1))
        x = paddle.matmul(x, t_net)
        x = paddle.transpose(x, (0, 2, 1))
        x = self.mlp_1(x)

        t_net = self.feature_transform_net(x)
        t_net = paddle.squeeze(t_net, axis=-1)
        t_net = self.feature_fc(t_net)
        t_net = paddle.reshape(t_net, [batchsize, 64, 64])

        x = paddle.squeeze(x, axis=-1)
        x = paddle.transpose(x, (0, 2, 1))
        x = paddle.matmul(x, t_net)
        x = paddle.transpose(x, (0, 2, 1))
        point_feat = x
        x = self.mlp_2(x)
        x = paddle.max(x, axis=-1, keepdim=True)
        global_feat_expand = paddle.tile(x, [1, 1, self.max_point])
        x = paddle.concat([point_feat, global_feat_expand], axis=1)
        x = self.seg_net(x)
        x = paddle.squeeze(x, axis=-1)
        x = paddle.transpose(x, (0, 2, 1))

        return x