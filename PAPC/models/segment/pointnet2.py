import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from PAPC.models.layers import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation, Categorical

class PointNet2_SSG_Seg(nn.Layer):
    def __init__(self, name_scope='PointNet2_SSG_Seg_', num_classes=16, num_parts=50, normal_channel=False):
        super(PointNet2_SSG_Seg, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.num_classes = num_classes
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6+additional_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+16+6+additional_channel, mlp=[128, 128, 128])
        self.conv1 = nn.Conv1D(128, 128, 1)
        self.bn1 = nn.BatchNorm1D(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1D(128, num_parts, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = paddle.tile(cls_label.reshape([B, self.num_classes, 1]), [1, 1, N])
        l0_points = self.fp1(l0_xyz, l1_xyz, paddle.concat([cls_label_one_hot, l0_xyz, l0_points], 1), l1_points)
        # FC layers
        feat =  F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)

        return x

class PointNet2_MSG_Seg(nn.Layer):
    def __init__(self, name_scope='PointNet2_MSG_Seg_', num_classes=16, num_parts=50, normal_channel=False):
        super(PointNet2_MSG_Seg, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.num_classes = num_classes
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1D(128, 128, 1)
        self.bn1 = nn.BatchNorm1D(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1D(128, num_parts, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = paddle.tile(cls_label.reshape([B, self.num_classes, 1]), [1, 1, N])
        l0_points = self.fp1(l0_xyz, l1_xyz, paddle.concat([cls_label_one_hot, l0_xyz, l0_points], axis=1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)

        return x

if __name__ == '__main__':
    model = PointNet2_SSG_Seg()
    paddle.summary(model, ((64, 3, 1024), (64, 16, 1)))
    model = PointNet2_MSG_Seg()
    paddle.summary(model, ((64, 3, 1024), (64, 16, 1)))