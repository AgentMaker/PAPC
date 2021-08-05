#coding=utf-8
"""
Code based on Wang GUOJUN.
Licensed under MIT License [see LICENSE].
"""

import sys
sys.path.append("..")
import os
import time
import numpy as np
import torch


import rospy
from ros_numpy import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from numpy.lib.recfunctions import structured_to_unstructured
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from pyquaternion import Quaternion
from visualization_msgs.msg import Marker,MarkerArray

import argparse
from torch2trt import TRTModule
from params.configs import cfg,cfg_from_yaml_file
from core import (build_target_assigner,build_anchor_generator,
                    build_voxel_generator,build_box_coder)
from libs.ops import box_np_ops 
from data.preprocess import voxel_padding
from models import build_network

class SecondModel:
    def __init__(self,
                 trt_dir,
                 weights_file,
                 config_path,
                 max_voxel_num = 12000,
                 tensorrt = True,
                 anchors_area= 0.01):
        
        self.trt_dir = trt_dir
        self.config_path = config_path
        self.anchors_area = anchors_area
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #load config

        config = cfg_from_yaml_file(config_path,cfg)
        self.config = config 
        self.model_cfg = config.MODEL 
        voxel_cfg = config.VOXEL_GENERATOR
        classes_cfg= config.TARGET_ASSIGNER.ANCHOR_GENERATORS
        
        #build generators
        self.anchor_generator = build_anchor_generator(classes_cfg[0])
        self.voxel_generator = build_voxel_generator(voxel_cfg)
        # build parameter
        self.voxel_size=self.voxel_generator.voxel_size
        self.grid_size=self.voxel_generator.grid_size
        self.pc_range =self.voxel_generator.point_cloud_range

        self.max_voxel_num=config.TRAIN_INPUT_READER.MAX_NUMBER_OF_VOXELS
        out_size_factor = self.model_cfg.BACKBONE.layer_strides[0] //self.model_cfg.BACKBONE.upsample_strides[0]

        feature_map_size = self.grid_size[:2] // out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1]
        self.anchors = self.anchor_generator.generate(feature_map_size).reshape((1, -1, 7))
        self.anchors_bv = box_np_ops.rbbox2d_to_near_bbox(self.anchors[0,:][:,[0, 1, 3, 4, 6]])
        
        # buld network
        net = self.build_network().to(self.device)
        
        #load 
        
        state_dict=torch.load(weights_file)
        net.load_state_dict(state_dict,strict = False)
        
        #use tensorrt
        
        if tensorrt:
            pfn_trt = TRTModule()
            pfn_trt.load_state_dict(torch.load(os.path.join(trt_dir,'pfn.trt')))
            rpn_trt = TRTModule()
            rpn_trt.load_state_dict(torch.load(os.path.join(trt_dir,'backbone.trt')))
            net.pfn = pfn_trt
            net.rpn = rpn_trt
            
        self.net=net.eval()
        
        
    def build_network(self):
        ######################
        # BUILD TARGET ASSIGNER
        ######################
        bv_range = self.voxel_generator.point_cloud_range[[0, 1, 3, 4]]
        box_coder = build_box_coder(self.config.BOX_CODER)
        target_assigner_cfg = self.config.TARGET_ASSIGNER
        target_assigner = build_target_assigner(target_assigner_cfg,
                                                    bv_range, box_coder) 
        ######################
        # BUILD NET
        ######################
        self.model_cfg.XAVIER = True 
        net = build_network(self.model_cfg, 
                                   self.voxel_generator, 
                                   target_assigner)       
        return net
            
        
    def predict(self,pointclouds):
        
        t0 = time.time()
        
        ret = self.voxel_generator.generate(pointclouds, max_voxels=self.max_voxel_num)
        voxels = ret[0]
        coords = ret[1]
        num_points = ret[2]
        voxels, num_points, coords, voxel_mask = voxel_padding(voxels, num_points,
                                                       coords, max_voxel_num=self.max_voxel_num)
        

        example = {
            "anchors": self.anchors,
            "voxels": voxels,
            "num_points": num_points,
            "coordinates": coords,
            'voxel_mask': voxel_mask,
            "metadata": [{"image_idx": '000000'}]}

        #build anchors mask

        if self.anchors_area >= 0:

            # 计算每个grid map坐标位置是否有pillars（非空）
            dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
                coords, tuple(self.grid_size[::-1][1:]))
            dense_voxel_map = dense_voxel_map.cumsum(0)
            dense_voxel_map = dense_voxel_map.cumsum(1)

            # 计算每个anchor_bev占有的非空的pillars
            anchors_area = box_np_ops.fused_get_anchors_area(
                dense_voxel_map, self.anchors_bv, self.voxel_size, self.pc_range, self.grid_size)
            anchors_mask = anchors_area >= anchors_area
            if anchors_mask.sum() < 1:
                anchors_mask = np.zeros(anchors_area.shape[0], dtype=np.bool)
                print("anchors_mask is zero")
            example['anchors_mask'] = anchors_mask.reshape(-1,1)
        # turn torch list
        example_list = example_to_tensorlist_with_batch(example, self.config, device = self.device)
        
        #inference
        with torch.no_grad():
            b1 = time.time()
            boxes,scores = self.net(example_list)[0][:2]
            spf = time.time()-b1
            try:
                boxes=boxes.detach().cpu().numpy()
                scores=scores.detach().cpu().numpy()
            except:
                pass
            print("current frame process time is {:.3f}ms".format((time.time()-t0)*1000))
            print('second/frame:{:3f}ms'.format(spf*1000))
        return boxes,scores
            
        


class SecondROS:
    def __init__(self,
                 trt_dir,
                 weights_file,
                 config_path,
                 is_tensorrt=True,
                 anchors_area =0.1):
        
        rospy.init_node("second_ros")

        print("TensorRT Engine: {}.".format(is_tensorrt))

        # Subscriber
        self.model = SecondModel(trt_dir=trt_dir,
                                 weights_file = weights_file,
                                 config_path = config_path,
                                 tensorrt = is_tensorrt,
                                 anchors_area = anchors_area)

        print("Waiting for ROS topic: /raw_cloud")
        self.sub_lidar = rospy.Subscriber("/raw_cloud",PointCloud2,self.lidar_callback,queue_size = 1)

        # Publisher
        self.pub_bbox = rospy.Publisher("/boxes", BoundingBoxArray, queue_size=1)
        self.pub_text=rospy.Publisher("/scores",MarkerArray,queue_size=0)
        self.pub_cloud = rospy.Publisher("/cloud_filtered", PointCloud2, queue_size=0)

        rospy.spin()

    
    def lidar_callback(self,msg):
        pc_arr=point_cloud2.pointcloud2_to_array(msg)
        pc_arr = structured_to_unstructured(pc_arr)
        #print(pc_arr.shape)
        pc_arr=pc_arr.reshape(-1,4)
        lidar_boxes,lidar_scores = self.model.predict(pc_arr)
        #print(lidar_boxes)
        # points.dtype=[('x', np.float32),('y', np.float32),('z', np.float32),('intensity', np.float32)]
        # cloud_msg=point_cloud2.array_to_pointcloud2(points,rospy.Time.now(),"rslidar")

        if lidar_boxes is not None:
            num_detects = len(lidar_boxes) #if len(lidar_boxes)<=10 else 10

            arr_bbox = BoundingBoxArray()
            arr_score=MarkerArray()
            
            for i in range(num_detects):
                bbox = BoundingBox()
                bbox.header.frame_id = msg.header.frame_id
                bbox.header.stamp = rospy.Time.now()

                bbox.pose.position.x = float(lidar_boxes[i][0])
                bbox.pose.position.y = float(lidar_boxes[i][1])
                #bbox.pose.position.z = float(lidar_boxes[i][2])
                bbox.pose.position.z = float(lidar_boxes[i][2]) + float(lidar_boxes[i][5]) / 2
                bbox.dimensions.x = float(lidar_boxes[i][3])  # width
                bbox.dimensions.y = float(lidar_boxes[i][4])  # length
                bbox.dimensions.z = float(lidar_boxes[i][5])  # height

                q = Quaternion(axis=(0, 0, 1), radians=float(-lidar_boxes[i][6]))
                bbox.pose.orientation.x = q.x
                bbox.pose.orientation.y = q.y
                bbox.pose.orientation.z = q.z
                bbox.pose.orientation.w = q.w

                arr_bbox.boxes.append(bbox)

                marker = Marker()
                marker.header.frame_id =msg.header.frame_id
                marker.header.stamp = rospy.Time.now()
                marker.ns = "basic_shapes"
                marker.id = i
                marker.type = Marker.TEXT_VIEW_FACING
                marker.action = Marker.ADD
                marker.lifetime=rospy.Duration(0.15)
                marker.scale.x = 4
                marker.scale.y = 4
                marker.scale.z = 4

                # Marker的颜色和透明度
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1
                marker.color.a = 1
                marker.pose.position.x=float(lidar_boxes[i][0])
                marker.pose.position.y = float(lidar_boxes[i][1])
                marker.pose.position.z = float(lidar_boxes[i][2]) + float(lidar_boxes[i][5]) / 2
                marker.text=str(np.around(lidar_scores[i],2))
                arr_score.markers.append(marker)
            arr_bbox.header.frame_id = msg.header.frame_id
            arr_bbox.header.stamp = rospy.Time.now()
            print("Number of detections: {}".format(num_detects))

            self.pub_bbox.publish(arr_bbox)
            self.pub_text.publish(arr_score)
            # self.pub_cloud.publish(cloud_msg)
    


    
    
    
    
# conver torch functions
def get_paddings_indicator_np(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = np.expand_dims(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = np.arange(max_num, dtype=np.int).reshape(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.astype(np.int32) > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator

def example_to_tensorlist_with_batch(example, config,device=None,float_type=torch.float32):
    example_list = [None] * 13
    pillar_x = example['voxels'][:, :, 0][np.newaxis,np.newaxis,:,:] # (1,K,T)
    pillar_y = example['voxels'][:, :, 1][np.newaxis,np.newaxis,:,:]
    pillar_z = example['voxels'][:, :, 2][np.newaxis,np.newaxis,:,:]
    pillar_i = example['voxels'][:, :, 3][np.newaxis,np.newaxis,:,:]
    num_points_per_pillar = example['num_points'][np.newaxis,:]  # (N,K,)
    #print(num_points_per_pillar.shape)
    coors = example['coordinates'] [np.newaxis,:]  # (N,K,3)
    anchors = example['anchors']  # (B,num_anchors,7)
    image_ids = [int(elem['image_idx']) for elem in example['metadata']]
    image_ids = np.array(image_ids, dtype=np.int32)
    voxel_mask = example['voxel_mask'][np.newaxis,:] # (N,K)
    # ################################################################
    # Find distance of x, y, z from pillar center
    coors_x = example['coordinates'][:, 2][np.newaxis,:]  # (N,K)
    coors_y = example['coordinates'][:, 1][np.newaxis,:]
    pc_range = cfg.MODEL.POST_PROCESSING.post_center_limit_range
    x_sub = coors_x[:, np.newaxis, :, np.newaxis] * cfg.VOXEL_GENERATOR.VOXEL_SIZE[0] +pc_range[0] # Pillars的中心的位置坐标 (N,1,K,1)
    y_sub = coors_y[:, np.newaxis, :, np.newaxis] * cfg.VOXEL_GENERATOR.VOXEL_SIZE[1] +pc_range[1] 
    # print("before repeat x_sub nan is ",torch.nonzero(torch.isnan(x_sub)).shape)
    # print("before repeat y_sub nan is ", torch.nonzero(torch.isnan(y_sub)).shape)

    x_sub_shaped = x_sub.repeat(pillar_x.shape[3], -1)
    y_sub_shaped = y_sub.repeat(pillar_x.shape[3], -1)  # (N,1,K,T)
    # print("after repeat x_sub nan is ", torch.nonzero(torch.isnan(x_sub_shaped)).shape)
    # print("after repeat y_sub nan is ", torch.nonzero(torch.isnan(y_sub_shaped)).shape)
    num_points_for_a_pillar = pillar_x.shape[3]  # (T)
    mask = get_paddings_indicator_np(num_points_per_pillar, num_points_for_a_pillar, axis=0)  # (N,T,K)
    mask = mask.transpose(0, 2, 1)  # (N,K,T)
    mask = mask[:, np.newaxis, :, :]  # (N,1,K,T)
    mask = mask.astype(pillar_x.dtype)

    example_list[0] = torch.tensor(pillar_x, dtype=float_type, device=device)
    example_list[1] = torch.tensor(pillar_y, dtype=float_type, device=device)
    example_list[2] = torch.tensor(pillar_z, dtype=float_type, device=device)
    example_list[3] = torch.tensor(pillar_i, dtype=float_type, device=device)
    example_list[4] = torch.tensor(num_points_per_pillar, dtype=float_type, device=device)
    example_list[5] = torch.tensor(x_sub_shaped, dtype=float_type, device=device)
    example_list[6] = torch.tensor(y_sub_shaped, dtype=float_type, device=device)
    example_list[7] = torch.tensor(mask, dtype=float_type, device=device)
    example_list[8] = torch.tensor(example['coordinates'], dtype=torch.int32, device=device)
    example_list[9] = torch.tensor(voxel_mask, dtype=torch.bool, device=device)
    example_list[10] = torch.tensor(anchors, dtype=float_type, device=device)
    example_list[11] = torch.tensor(image_ids, dtype=torch.int32, device=device)
    if 'anchors_mask' in example.keys():
        example_list[12]=torch.tensor(example['anchors_mask'], dtype=torch.bool, device=device)
        #print(example_list[12])
    else:
        example_list[12]=None
    return example_list





if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--weights_file", type=str, default="./params/weights/pointpillars/PointPillars.pdparams")
    parse.add_argument("--config_path",type=str,default="./params/configs/pointpillars_kitti_car_xy16.yaml")
    parse.add_argument("--trt_dir",type=str,default="./params/TensorRT/pointpillar_0827")
    parse.add_argument("--anchors_area", type=int, default=0.001)

    args=parse.parse_args()



    second_ros=SecondROS(trt_dir=args.trt_dir,
                         weights_file = args.weights_file,
                         config_path=args.config_path,
                         is_tensorrt=True,
                         anchors_area=args.anchors_area)

