"""
Code based on Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""


import time
from enum import Enum
from functools import reduce

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import libs 
from libs.tools import metrics


from models.bones.pillars import (PillarFeatureNet, PointPillarsScatter)
from models.bones.rpn import RPN

from core.losses import (WeightedSigmoidClassificationLoss,
                        WeightedSmoothL1LocalizationLoss,
                        WeightedSoftmaxClassificationLoss)
from libs.ops import box_np_ops, box_paddle_ops

class PointPillars(nn.Layer):
    def __init__(self,
                output_shape,
                model_cfg,
                target_assigner):
        super().__init__()

        self.name =model_cfg.NAME
        self._pc_range = model_cfg.POINT_CLOUD_RANGE
        self._voxel_size = model_cfg.GRID_SIZE
        self._num_class = model_cfg.NUM_CLASS
        self._use_bev = model_cfg.BACKBONE.use_bev
        self._total_forward_time = 0.0
        self._total_postprocess_time = 0.0
        self._total_inference_count = 0
        
        #for prepare loss weights
        self._pos_cls_weight = model_cfg.pos_cls_weight 
        self._neg_cls_weight = model_cfg.neg_cls_weight 
        self._loss_norm_type = model_cfg.loss_norm_type
        #for create loss 
        self._loc_loss_ftor = model_cfg.loc_loss_ftor
        self._cls_loss_ftor = model_cfg.cls_loss_ftor
        self._dir_loss_ftor = WeightedSoftmaxClassificationLoss()
        self._direction_loss_weight = model_cfg.LOSS.direction_loss_weight
        self._encode_rad_error_by_sin = model_cfg.ENCODE_RAD_ERROR_BY_SIN
        #
        self._cls_loss_weight = model_cfg.cls_weight
        self._loc_loss_weight = model_cfg.loc_weight
        # for direction classifier 
        self._use_direction_classifier = model_cfg.BACKBONE.use_direction_classifier
        # for predict
        self._use_sigmoid_score = model_cfg.POST_PROCESSING.use_sigmoid_score
        self._box_coder = target_assigner.box_coder 
        self.target_assigner = target_assigner
        #for nms 
        self._multiclass_nms = model_cfg.PREDICT.multiclass_nms
        self._use_rotate_nms = model_cfg.PREDICT.use_rotate_nms

        self._nms_score_threshold = model_cfg.POST_PROCESSING.nms_score_threshold
        self._nms_pre_max_size = model_cfg.POST_PROCESSING.nms_pre_max_size
        self._nms_post_max_size = model_cfg.POST_PROCESSING.nms_post_max_size
        self._nms_iou_threshold = model_cfg.POST_PROCESSING.nms_iou_threshold
        self._use_sigmoid_score = model_cfg.POST_PROCESSING.use_sigmoid_score
        # self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros=model_cfg.BACKBONE.encode_background_as_zeros
        #1.PFN
        self.pfn = PillarFeatureNet(model_cfg.num_input_features,
                                    model_cfg.PILLAR_FEATURE_EXTRACTOR.use_norm,
                                    num_filters=model_cfg.pfn_num_filters,
                                    with_distance=model_cfg.PILLAR_FEATURE_EXTRACTOR.with_distance,
                                    voxel_size = self._voxel_size,
                                    pc_range = self._pc_range)

        #2.sparse middle
        self.mfe = PointPillarsScatter(output_shape=output_shape,
                                       num_input_features=model_cfg.pfn_num_filters[-1])
        num_rpn_input_filters=self.mfe.nchannels
        #3.rpn
        self.rpn =  RPN(
                        use_norm=model_cfg.BACKBONE.use_norm,
                        num_class=self._num_class,
                        layer_nums=model_cfg.BACKBONE.layer_nums,
                        layer_strides=model_cfg.BACKBONE.layer_strides,
                        num_filters=model_cfg.BACKBONE.num_filters,
                        upsample_strides=model_cfg.BACKBONE.upsample_strides,
                        num_upsample_filters=model_cfg.BACKBONE.num_upsample_filters,
                        num_input_filters=num_rpn_input_filters,
                        num_anchor_per_loc=target_assigner.num_anchors_per_location,
                        encode_background_as_zeros=self._encode_background_as_zeros,
                        use_direction_classifier=model_cfg.BACKBONE.use_direction_classifier,
                        use_bev=model_cfg.BACKBONE.use_bev,
                        use_groupnorm=model_cfg.BACKBONE.use_groupnorm,
                        num_groups=model_cfg.BACKBONE.num_groups,
                        box_code_size=target_assigner.box_coder.code_size
                        )
        self.rpn_acc = metrics.Accuracy(
            dim=-1, encode_background_as_zeros=self._encode_background_as_zeros)
        self.rpn_precision = metrics.Precision(dim=-1)
        self.rpn_recall = metrics.Recall(dim=-1)
        self.rpn_metrics = metrics.PrecisionRecall(
            dim=-1,
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            use_sigmoid_score=self._use_sigmoid_score,
            encode_background_as_zeros=self._encode_background_as_zeros)
        
        self.rpn_cls_loss =metrics.Scalar()
        self.rpn_loc_loss =metrics.Scalar()
        self.rpn_total_loss = metrics.Scalar()
        self.register_buffer("global_step", paddle.to_tensor(0).astype("int64"))

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.numpy()[0])

        
    def forward(self, example):
        # print("++++++++++++++++++++++++++++++++++++START FORWARD++++++++++++++++++++++++++++++++++++++++++++++++")
        voxels = example['voxels']
        num_points = example['num_points']
        coors = example["coordinates"]
        batch_anchors = example["anchors"]        
        batch_size_dev = batch_anchors.shape[0]
        t = time.time()
        # features: [num_voxels, max_num_points_per_voxel, 7]
        # num_points: [num_voxels]
        # coors: [num_voxels, 4]
        # print("++++++++++++++++++++++++++++++++++++START PFN++++++++++++++++++++++++++++++++++++++++++++++++")
        voxel_features = self.pfn(voxels, num_points, coors)
        # print("++++++++++++++++++++++++++++++++++++START MFE++++++++++++++++++++++++++++++++++++++++++++++++")
        spatial_features = self.mfe(voxel_features,coors,batch_size_dev)
        # print("++++++++++++++++++++++++++++++++++++START RPN++++++++++++++++++++++++++++++++++++++++++++++++")
        if self._use_bev:
            preds_dict = self.rpn(spatial_features,example['bev_map'])
        else:
            preds_dict = self.rpn(spatial_features)
            # print("++++++++++++++++++++++++++++++++++++OVER RPN++++++++++++++++++++++++++++++++++++++++++++++++")
        box_preds = preds_dict["box_preds"]
        cls_preds = preds_dict["cls_preds"]
        self._total_forward_time += time.time() - t
        if self.training:
            labels = example['labels']
            reg_targets = example['reg_targets']

            # print("++++++++++++++++++++++++++++++++++++START PRE_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")
            cls_weights, reg_weights, cared = prepare_loss_weights(
                labels,
                pos_cls_weight=self._pos_cls_weight,
                neg_cls_weight=self._neg_cls_weight,
                loss_norm_type=self._loss_norm_type,
                dtype=voxels.dtype)
            # print("++++++++++++++++++++++++++++++++++++OVER PRE_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")
            cls_targets = labels * cared.astype(labels.dtype)
            cls_targets = cls_targets.unsqueeze(-1)
            # print("++++++++++++++++++++++++++++++++++++OVER PRE_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")

            # print("++++++++++++++++++++++++++++++++++++START CREATE_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")
            loc_loss,cls_loss = create_loss(
                self._loc_loss_ftor,
                self._cls_loss_ftor,
                box_preds=box_preds,
                cls_preds=cls_preds,
                cls_targets=cls_targets,
                cls_weights=cls_weights,
                reg_targets=reg_targets,
                reg_weights=reg_weights,
                num_class=self._num_class,
                encode_rad_error_by_sin=self._encode_rad_error_by_sin,
                encode_background_as_zeros=self._encode_background_as_zeros,
                box_code_size=self._box_coder.code_size,)
            # print("++++++++++++++++++++++++++++++++++++OVER CREATE_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")

            loc_loss_reduced = loc_loss.sum() / batch_size_dev
            loc_loss_reduced *= self._loc_loss_weight
            cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
            cls_pos_loss /= self._pos_cls_weight
            cls_neg_loss /= self._neg_cls_weight
            cls_loss_reduced = cls_loss.sum() / batch_size_dev
            cls_loss_reduced *= self._cls_loss_weight
            loss = loc_loss_reduced + cls_loss_reduced
            # print("++++++++++++++++++++++++++++++++++++OVER CAL_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")
            if self._use_direction_classifier:
                # print("++++++++++++++++++++++++++++++++++++START CLAS_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")
                dir_targets = get_direction_target(example['anchors'],
                                                   reg_targets)
                dir_logits = preds_dict["dir_cls_preds"].reshape((batch_size_dev, -1, 2))
                weights = (labels > 0).astype(dir_logits.dtype)
                weights /= paddle.clip(weights.sum(-1, keepdim=True), min=1.0)
                dir_loss = self._dir_loss_ftor(dir_logits, dir_targets, weights=weights)
                dir_loss = dir_loss.sum() / batch_size_dev
                loss += dir_loss * self._direction_loss_weight
                # print("++++++++++++++++++++++++++++++++++++OVER CLAS_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")

            return {
                "loss": loss,
                "cls_loss": cls_loss,
                "loc_loss": loc_loss,
                "cls_pos_loss": cls_pos_loss,
                "cls_neg_loss": cls_neg_loss,
                "cls_preds": cls_preds,
                "dir_loss_reduced": dir_loss,
                "cls_loss_reduced": cls_loss_reduced,
                "loc_loss_reduced": loc_loss_reduced,
                "cared": cared,
            }
        else:
            # print("++++++++++++++++++++++++++++++++++++OVER EVAL_PREDICT++++++++++++++++++++++++++++++++++++++++++++++++")
            return self.predict(example, preds_dict)
    
    def predict(self, example, preds_dict):
        t = time.time()
        batch_size = example['anchors'].shape[0]
        batch_anchors = example["anchors"].reshape((batch_size, -1, 7))

        self._total_inference_count += batch_size
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].reshape((batch_size, -1))
        batch_imgidx = example['image_idx']

        self._total_forward_time += time.time() - t
        t = time.time()

        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.reshape((batch_size, -1, self._box_coder.code_size))             

        num_class_with_bg = self._num_class
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_class + 1
        batch_cls_preds = batch_cls_preds.reshape((batch_size, -1, num_class_with_bg))
        batch_box_preds = self._box_coder.decode_paddle(batch_box_preds,
                                                       batch_anchors)
        if self._use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.reshape((batch_size, -1, 2))
        else:
            batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        for box_preds, cls_preds, dir_preds, img_idx, a_mask in zip(batch_box_preds, batch_cls_preds, batch_dir_preds, batch_imgidx, batch_anchors_mask.astype("int64")):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            if self._use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                dir_labels = paddle.argmax(dir_preds, axis=-1)[1]
            if self._encode_background_as_zeros:
                # this don't support softmax
                assert self._use_sigmoid_score is True
                total_scores = F.sigmoid(cls_preds)
            else:
                # encode background as first element in one-hot vector
                if self._use_sigmoid_score:
                    total_scores = F.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, axis=-1)[..., 1:]
            
            #apply nms in birdeye reshape
            if self._use_rotate_nms:
                nms_func = box_paddle_ops.rotate_nms
            else:
                nms_func = box_paddle_ops.nms
            selected_boxes = None
            selected_labels = None
            selected_scores = None
            selected_dir_labels = None

            if self._multiclass_nms:
                #curently only support class-agnostic boxes.
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not self._use_rotate_nms:
                    box_preds_corners = box_paddle_ops.center_to_corner_box2d(
                        boxes_for_nms[:,:2],boxes_for_nms[:,2:4],
                        boxes_for_nms[:,4]
                    )
                    box_for_nms = box_paddle_ops.corner_to_standup_nd(box_preds_corners)
                boxes_for_mcnms = boxes_for_nms.unsqueeze(1)
                selected_per_class = box_paddle_ops.multiclass_nms(
                    nms_func=nms_func,
                    boxes=boxes_for_mcnms,
                    scores=total_scores,
                    num_class=self._num_class,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                    iou_threshold=self._nms_iou_threshold,
                    score_thresh=self._nms_score_threshold,
                )
                selected_boxes, selected_labels, selected_scores = [] , [], []
                selected_dir_labels = []
                for i , selected in enumerate(selected_per_class):
                    if selected is not None:
                        num_dets = selected.shape[0]
                        selected_boxes.append(box_preds[selected])
                        selected_labels.append(
                            paddle.full([num_dets], i, dtype=paddle.int64)
                        )
                        if len(selected_boxes) > 0:
                            selected_boxes = paddle.concat(selected_boxes, axis=0)
                            selected_labels = paddle.concat(selected_labels, axis=0)
                            selected_scores = paddle.concat(selected_scores, axis=0)
                            if self._use_direction_classifier:
                                selected_dir_labels = paddle.concat(selected_dir_labels, axis=0)
                else:
                    selected_boxes = None
                    selected_labels = None
                    selected_scores = None
                    selected_dir_labels = None
            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box
                if num_class_with_bg ==1:
                    top_scores = total_scores.squeeze(-1)
                    top_labels = paddle.zeros(total_scores.shape[0], dtype=paddle.int64)
                else:
                    top_scores, top_labels = paddle.max(total_scores, axis=-1)

                if self._nms_score_threshold > 0.0:
                    thresh = paddle.to_tensor([self._nms_score_threshold]).astype(total_scores.dtype)
                    top_scores_keep = (top_scores >= thresh)
                    top_scores = top_scores.masked_select(top_scores_keep)
                if top_scores.shape[0] != 0:
                    if self._nms_score_threshold > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self._use_direction_classifier:
                            dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:,[0,1,3,4,6]]
                    if not self._use_rotate_nms:
                        box_preds_corners = box_paddle_ops.center_to_corner_box2d(
                            boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                            boxes_for_nms[:, 4])
                        boxes_for_nms = box_paddle_ops.corner_to_standup_nd(
                            box_preds_corners)
                    # the nms in 3d detection just remove overlap boxes.
                    selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=self._nms_pre_max_size,
                        post_max_size=self._nms_post_max_size,
                        iou_threshold=self._nms_iou_threshold,
                    )
                else:
                    selected = None
                if selected is not None:
                    selected_boxes = box_preds[selected]
                    if self._use_direction_classifier:
                        selected_dir_labels = dir_labels[selected]
                    selected_labels = top_labels[selected]
                    selected_scores = top_scores[selected]

            #finally generate predictions.
            if selected_boxes is not None:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self._use_direction_classifier:
                    dir_labels = selected_dir_labels
                    opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.bool()
                    box_preds[...,-1] += paddle.where(
                        opp_labels,
                        paddle.to_tensor(np.pi).astype(box_preds),
                        paddle.to_tensor(0.0).astype(box_preds)
                    )
                final_box_preds = box_preds 
                final_scores = scores 
                final_labels = label_preds
                # predictions
                predictions_dict = {
                    "bbox": None,
                    "box3d_camera": None ,
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": final_labels,
                    "image_idx": img_idx,
                }
            else:
                predictions_dict = {
                    "bbox": None,
                    "box3d_camera": None,
                    "box3d_lidar": None,
                    "scores": None,
                    "label_preds": None,
                    "image_idx": img_idx,
                }
            predictions_dicts.append(predictions_dict)
        self._total_postprocess_time += time.time() - t
        return predictions_dicts

    @property
    def avg_forward_time(self):
        return self._total_forward_time / self._total_inference_count

    @property
    def avg_postprocess_time(self):
        return self._total_postprocess_time / self._total_inference_count

    def clear_time_metrics(self):
        self._total_forward_time = 0.0
        self._total_postprocess_time = 0.0
        self._total_inference_count = 0

    def metrics_to_float(self):
        self.rpn_acc.float()
        self.rpn_metrics.float()
        self.rpn_cls_loss.float()
        self.rpn_loc_loss.float()
        self.rpn_total_loss.float()
    
    def update_metrics(self,
                       cls_loss,
                       loc_loss,
                       cls_preds,
                       labels,
                       sampled):
        batch_size = cls_preds.shape[0]
        num_class = self._num_class
        if not self._encode_background_as_zeros:
            num_class += 1
        cls_preds = cls_preds.reshape((batch_size, -1, num_class))
        # print("++++++++++++++++++++++++++++++++++++START RPN_ACC++++++++++++++++++++++++++++++++++++++++++++++++")
        rpn_acc = self.rpn_acc(labels, cls_preds, sampled).numpy()[0]
        # print("++++++++++++++++++++++++++++++++++++OVER RPB_ACC++++++++++++++++++++++++++++++++++++++++++++++++")
        prec, recall = self.rpn_metrics(labels, cls_preds, sampled)
        # print("++++++++++++++++++++++++++++++++++++OVER RPN_METRICS++++++++++++++++++++++++++++++++++++++++++++++++")
        prec = prec.numpy()
        recall = recall.numpy()
        rpn_cls_loss = self.rpn_cls_loss(cls_loss).numpy()[0]
        rpn_loc_loss = self.rpn_loc_loss(loc_loss).numpy()[0]
        ret = {
            "cls_loss": float(rpn_cls_loss),
            "cls_loss_rt": float(cls_loss.numpy()[0]),
            'loc_loss': float(rpn_loc_loss),
            "loc_loss_rt": float(loc_loss.numpy()[0]),
            "rpn_acc": float(rpn_acc),
        }
        for i, thresh in enumerate(self.rpn_metrics.thresholds):
            ret[f"prec@{int(thresh*100)}"] = float(prec[i])
            ret[f"rec@{int(thresh*100)}"] = float(recall[i])
        return ret
    
    def clear_metrics(self):
        self.rpn_acc.clear()
        self.rpn_metrics.clear()
        self.rpn_cls_loss.clear()
        self.rpn_loc_loss.clear()
        self.rpn_total_loss.clear()

    @staticmethod
    def convert_norm_to_float(net):
        if isinstance(net, paddle.nn.BatchNorm):
            net.float()
        for child in net.children():
            PointPillars.convert_norm_to_float(net)
        return net


def prepare_loss_weights(labels,
                         pos_cls_weight= 1.0,
                         neg_cls_weight= 1.0,
                         loss_norm_type= 'NormByNumPositives',
                         dtype= paddle.float32):
    # print("++++++++++++++++++++++++++++++++++++START IN PRE_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")

    cared = labels >= 0
    #cared : [N,num_anchors]
    positives = labels >0 
    negatives = labels ==0
    negatives_cls_weights = negatives.astype(dtype) * neg_cls_weight
    cls_weights = neg_cls_weight + pos_cls_weight * positives.astype(dtype)
    reg_weights = positives.astype(dtype)
    # print("++++++++++++++++++++++++++++++++++++START IN PRE_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")
    if loss_norm_type == 'NormByNumExamples':
        num_examples = cared.astype(dtype).sum(1, keepdim=True)
        num_examples = paddle.clip(num_examples, min=1.0)
        cls_weights /= num_examples
        bbox_normalizer = positives.sum(1, keepdim=True).astype(dtype)
        reg_weights /= paddle.clip(bbox_normalizer, min=1.0)
    elif loss_norm_type == 'NormByNumPositives':  # for focal loss
        pos_normalizer = positives.astype(dtype).sum(1, keepdim=True)
        reg_weights /= paddle.clip(pos_normalizer, min=1.0)
        cls_weights /= paddle.clip(pos_normalizer, min=1.0)
    elif loss_norm_type == 'NormByNumPosNeg':
        pos_neg = paddle.stack([positives, negatives], axis=-1).astype(dtype)
        normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
        cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
        cls_normalizer = paddle.clip(cls_normalizer, min=1.0)
        # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
        normalizer = paddle.clip(normalizer, min=1.0)
        reg_weights /= normalizer[:, 0:1, 0]
        cls_weights /= cls_normalizer
    else:
        raise ValueError(
            f"unknown loss norm type.")
    # print("++++++++++++++++++++++++++++++++++++START IN PRE_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")
    return cls_weights, reg_weights, cared

def create_loss(loc_loss_ftor,
                cls_loss_ftor,
                box_preds,
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                encode_background_as_zeros=True,
                encode_rad_error_by_sin=True,
                box_code_size = 7):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.reshape((batch_size, -1, box_code_size))
    if encode_background_as_zeros:
        cls_preds = cls_preds.reshape((batch_size,-1,num_class))
    else:
        cls_preds = cls_preds.reshape((batch_size,-1,num_class+1))
    cls_targets = cls_targets.squeeze(-1)
    # print("++++++++++++++++++++++++++++++++++++START ONE_HOT++++++++++++++++++++++++++++++++++++++++++++++++")
    one_hot_targets = libs.tools.one_hot(
        cls_targets, depth=num_class+1,dtype=box_preds.dtype
    )
    # print("++++++++++++++++++++++++++++++++++++OVER ONE_HOT++++++++++++++++++++++++++++++++++++++++++++++++")
    if encode_background_as_zeros:
        # print("++++++++++++++++++++++++++++++++++++START ONE_HOT_CHANS++++++++++++++++++++++++++++++++++++++++++++++++")
        one_hot_targets = one_hot_targets[:, :, 1:]
        # print("++++++++++++++++++++++++++++++++++++OVER ONE_HOT_CHANS++++++++++++++++++++++++++++++++++++++++++++++++")
    if encode_rad_error_by_sin:
        # sin(a-b) = sina*cosb-cosa*sinb
        # print("++++++++++++++++++++++++++++++++++++START SIN++++++++++++++++++++++++++++++++++++++++++++++++")
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
        # print("++++++++++++++++++++++++++++++++++++START SIN++++++++++++++++++++++++++++++++++++++++++++++++")
    # print("++++++++++++++++++++++++++++++++++++START LOC_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")
    loc_losses = loc_loss_ftor(
        box_preds, reg_targets, weights=reg_weights)  # [N, M]
    # print("++++++++++++++++++++++++++++++++++++OVER LOC_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")
    # print("++++++++++++++++++++++++++++++++++++START CLAS_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
    # print("++++++++++++++++++++++++++++++++++++OVER CLAS_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")
    return loc_losses, cls_losses

def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = paddle.sin(boxes1[:, :, -1:]) * paddle.cos(
        boxes2[:, :, -1:])
    rad_tg_encoding = paddle.cos(boxes1[:, :, -1:]) * paddle.sin(boxes2[:, :, -1:])
    boxes1 = paddle.concat([boxes1[:, :, :-1], rad_pred_encoding], axis=-1)
    boxes2 = paddle.concat([boxes2[:, :, :-1], rad_tg_encoding], axis=-1)
    return boxes1, boxes2

def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    # print("++++++++++++++++++++++++++++++++++++START POS_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).astype(cls_loss.dtype) * cls_loss.reshape((batch_size, -1))
        cls_neg_loss = (labels == 0).astype(cls_loss.dtype) * cls_loss.reshape((batch_size, -1))
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[:, :, 1:].sum() / batch_size
        cls_neg_loss = cls_loss[:, :, 0].sum() / batch_size
    # print("++++++++++++++++++++++++++++++++++++OVER POS_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")
    return cls_pos_loss, cls_neg_loss

def get_direction_target(anchors, reg_targets, one_hot=True):
    # print("++++++++++++++++++++++++++++++++++++START DIRE_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")
    batch_size = reg_targets.shape[0]
    anchors = anchors.reshape((batch_size, -1, 7))
    rot_gt = reg_targets[:, :, -1] + anchors[:, :, -1]
    dir_cls_targets = (rot_gt > 0).astype("int64")
    if one_hot:
        dir_cls_targets = libs.tools.one_hot(
            dir_cls_targets, 2, dtype=anchors.dtype)
    # print("++++++++++++++++++++++++++++++++++++OVER DIRE_LOSS++++++++++++++++++++++++++++++++++++++++++++++++")
    return dir_cls_targets
