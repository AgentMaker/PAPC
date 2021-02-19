import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


def Categorical(y, num_class=16):
    """
        1-hot encodes a tensor
    """
    new_y = np.eye(num_class)[y,]
    new_y = paddle.to_tensor(new_y).transpose([0, 2, 1]).astype('float32')

    return new_y


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per - point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * paddle.matmul(src, dst.transpose([0, 2, 1]))
    dist += paddle.sum(src ** 2, axis=-1).reshape([B, N, 1])
    dist += paddle.sum(dst ** 2, axis=-1).reshape([B, 1, M])

    return dist


def index_points(points, idx):
    """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = paddle.tile(paddle.arange(B).reshape(view_shape), repeat_shape)
    points_np = points.numpy()
    batch_indices_np = batch_indices.numpy().astype('int64')
    idx_np = idx.numpy().astype('int64')
    new_points = paddle.to_tensor(points_np[batch_indices_np, idx_np, :])

    return new_points


def farthest_point_sample(xyz, npoint):
    """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.shape
    centroids = paddle.zeros([B, npoint])
    distance = paddle.ones([B, N])
    farthest = paddle.randint(0, N, (B,))
    batch_indices = paddle.arange(B)

    for i in range(npoint):
        centroids[:, i] = farthest
        xyz_np = xyz.numpy()
        batch_indices_np = batch_indices.numpy().astype('int64')
        farthest_np = farthest.numpy().astype('int64')
        centroid = xyz_np[batch_indices_np, farthest_np, :]
        centroid = paddle.to_tensor(centroid).unsqueeze(1)
        dist = paddle.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance_np = distance.numpy()
        dist_np = dist.numpy()
        mask_np = mask.numpy()
        distance_np[mask_np] = dist_np[mask_np]
        distance = paddle.to_tensor(distance_np)
        farthest = paddle.argmax(distance, -1)

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
        Input:
            radius: local region radius
            nsample: max sample number in local region
            xyz: all points, [B, N, 3]
            new_xyz: query points, [B, S, 3]
        Return:
            group_idx: grouped points index, [B, S, nsample]
    """
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = paddle.tile(paddle.arange(N).reshape([1, 1, N]), [B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    mask = sqrdists > radius ** 2
    group_idx_np = group_idx.numpy()
    mask_np = mask.numpy()
    group_idx_np[mask_np] = N
    group_idx = paddle.to_tensor(group_idx_np)
    group_idx = group_idx.sort(axis=-1)[:, :, :nsample]
    group_first = paddle.tile(group_idx[:, :, 0].reshape([B, S, 1]), [1, 1, nsample])
    mask = group_idx == N
    group_idx_np = group_idx.numpy()
    group_first_np = group_first.numpy()
    mask_np = mask.numpy()
    group_idx_np[mask_np] = group_first_np[mask_np]
    group_idx = paddle.to_tensor(group_idx_np)

    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
        Input:
            npoint:
            radius:
            nsample:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, npoint, nsample, 3]
            new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.reshape([B, S, 1, C])

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = paddle.concat([grouped_xyz_norm, grouped_points], axis=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
        Input:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, 1, 3]
            new_points: sampled points data, [B, 1, N, 3+D]
    """
    B, N, C = xyz.shape
    new_xyz = paddle.zeros([B, 1, C])
    grouped_xyz = xyz.reshape([B, 1, N, C])
    if points is not None:
        new_points = paddle.concat([grouped_xyz, points.reshape([B, 1, N, -1])], axis=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Layer):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = []
        self.mlp_bns = []
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2D(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2D(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
            Input:
                xyz: input points position data, [B, C, N]
                points: input points data, [B, D, N]
            Return:
                new_xyz: sampled points position data, [B, C, S]
                new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.transpose([0, 2, 1])
        if points is not None:
            points = points.transpose([0, 2, 1])

        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.transpose([0, 3, 2, 1])
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = paddle.max(new_points, 2)
        new_xyz = new_xyz.transpose([0, 2, 1])
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Layer):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = []
        self.bn_blocks = []
        for i in range(len(mlp_list)):
            convs = []
            bns = []
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2D(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2D(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.transpose([0, 2, 1])
        if points is not None:
            points = points.transpose([0, 2, 1])

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.reshape([B, S, 1, C])
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = paddle.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.transpose([0, 3, 2, 1])
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = paddle.max(grouped_points, 2)  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.transpose([0, 2, 1])
        new_points_concat = paddle.concat(new_points_list, axis=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Layer):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = []
        self.mlp_bns = []
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1D(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1D(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.transpose([0, 2, 1])
        xyz2 = xyz2.transpose([0, 2, 1])

        points2 = points2.transpose([0, 2, 1])
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = paddle.tile(points2, [1, N, 1])
        else:
            dists = square_distance(xyz1, xyz2)
            dists = paddle.sort(dists, axis=-1)
            idx = paddle.argsort(dists, axis=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = paddle.sum(dist_recip, axis=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = paddle.sum(index_points(points2, idx) * weight.reshape([B, N, 3, 1]), axis=2)

        if points1 is not None:
            points1 = points1.transpose([0, 2, 1])
            new_points = paddle.concat([points1, interpolated_points], axis=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.transpose([0, 2, 1])
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points