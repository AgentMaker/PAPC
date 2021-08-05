import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from libs.nn import Empty
from libs.functional import mask_select, select_change
from libs.tools import get_paddings_indicator

class PFNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super(PFNLayer, self).__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:
            self.linear = nn.Linear(in_features=in_channels, out_features=out_channels, bias_attr=False)
            self.norm = nn.BatchNorm1D(num_features=out_channels, momentum=0.01, epsilon=1e-3)
        else:
            self.linear = nn.Linear(in_features=in_channels, out_features=out_channels, bias_attr=True)
            self.norm = Empty(num_features=out_channels)

    def forward(self, inputs):
        x = self.linear(inputs)
        x = self.norm(x.transpose((0, 2, 1))).transpose((0, 2, 1))
        x = F.relu(x)

        x_max = paddle.max(x, axis=1, keepdim=True)

        if self.last_vfe:
            return x_max
        else:
            x_repeat = paddle.tile(x_max, (1, inputs.shape[1], 1))
            x_concatenated = paddle.concat([x, x_repeat], axis=2)
            return x_concatenated

class PillarFeatureNet(nn.Layer):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,128),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):

        super().__init__()
        self.name = 'PillarFeatureNet'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.LayerList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):

        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(axis=1, keepdim=True) / num_voxels.astype(features.dtype).reshape((-1, 1, 1))
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = paddle.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].astype("float32").unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].astype("float32").unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = paddle.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = paddle.concat(features_ls, axis=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = paddle.unsqueeze(mask, -1).astype(features.dtype)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()

class PointPillarsScatter(nn.Layer):
    def __init__(self,
                 output_shape,
                 num_input_features=4):
        super(PointPillarsScatter, self).__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size):
        batch_canvas = []
        for batch_itt in range(batch_size):
            canvas = paddle.zeros((self.nchannels, self.nx * self.ny), dtype=voxel_features.dtype)

            batch_mask = coords[:, 0] == batch_itt
            if batch_mask.any().numpy()[0] == True:
                this_coords = mask_select(coords, batch_mask)
                indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
                indices = indices.astype("int64")
                voxels = mask_select(voxel_features, batch_mask)
                voxels = voxels.t()

                canvas = select_change(canvas, voxels, indices)
            else:
                pass
            batch_canvas.append(canvas)

        batch_canvas = paddle.stack(batch_canvas, 0)
        batch_canvas = batch_canvas.reshape((batch_size, self.nchannels, self.ny, self.nx))

        return batch_canvas


if __name__ == '__main__':
    max_voxel_num = 120
    features = paddle.ones([max_voxel_num, 100, 4], dtype=paddle.fluid.core.VarDesc.VarType.FP32)
    num_voxels = paddle.ones([max_voxel_num], dtype=paddle.fluid.core.VarDesc.VarType.FP32)
    coors = paddle.ones([max_voxel_num, 100], dtype=paddle.fluid.core.VarDesc.VarType.FP32)

    PFN = PillarFeatureNet()
    res = PFN(features, num_voxels, coors)
    print(res.shape)