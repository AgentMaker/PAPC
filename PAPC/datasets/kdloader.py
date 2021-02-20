import os
import numpy as np
import random
import h5py
from PAPC.datasets.tools.build_KDTree import build_ClasKDTree, build_SegKDTree
from PAPC.datasets.datalist import *

def KDClasDataLoader(max_point=1024, batchsize=64, path='./data/', mode='train'):
    global train_list, test_list, val_list
    levels = (np.log(max_point) / np.log(2)).astype(int)

    datas = []
    split_dims_v = []
    points_v = []
    labels = []
    if mode == 'train':
        for file_list in train_list:
            f = h5py.File(os.path.join(path, file_list), 'r')
            datas.extend(f['data'][:, :max_point, :])
            labels.extend(f['label'])
            f.close()
    elif mode == 'test':
        for file_list in test_list:
            f = h5py.File(os.path.join(path, file_list), 'r')
            datas.extend(f['data'][:, :max_point, :])
            labels.extend(f['label'])
            f.close()
    else:
        for file_list in val_list:
            f = h5py.File(os.path.join(path, file_list), 'r')
            datas.extend(f['data'][:, :max_point, :])
            labels.extend(f['label'])
            f.close()
    datas = np.array(datas)

    for i in range(len(datas)):
        split_dim, tree = build_ClasKDTree(datas[i], depth=levels)
        split_dim_v = [np.array(item).astype(np.int64) for item in split_dim]
        split_dims_v.append(split_dim_v)
        points_v.append(tree[-1].transpose(0, 2, 1))

    split_dims_v = np.array(split_dims_v)
    points_v = np.array(points_v)
    labels = np.array(labels)
    print('==========load over==========')

    index_list = list(range(len(datas)))

    def KDClasDataGenerator():
        if mode == 'train':
            random.shuffle(index_list)
        for i in index_list:
            split_dim_v = split_dims_v[i]
            point_v = points_v[i].astype('float32')
            label = np.reshape(labels[i], [1, -1]).astype('int64')
            yield [point_v, split_dim_v], label

    return KDClasDataGenerator

def KDSegDataLoader(max_point=1024, batchsize=64, path='./data/', mode='train'):
    levels = (np.log(max_point) / np.log(2)).astype(int)

    datas = []
    split_dims_v = []
    points_v = []
    labels = []
    labels_v = []
    if mode == 'train':
        for file_list in train_list:
            f = h5py.File(os.path.join(path, file_list), 'r')
            datas.extend(f['data'][:, :max_point, :])
            labels.extend(f['pid'][:, :max_point])
            f.close()
    elif mode == 'test':
        for file_list in test_list:
            f = h5py.File(os.path.join(path, file_list), 'r')
            datas.extend(f['data'][:, :max_point, :])
            labels.extend(f['pid'][:, :max_point])
            f.close()
    else:
        for file_list in val_list:
            f = h5py.File(os.path.join(path, file_list), 'r')
            datas.extend(f['data'][:, :max_point, :])
            labels.extend(f['pid'][:, :max_point])
            f.close()
    datas = np.array(datas)

    for i in range(len(datas)):
        split_dim, point_tree, label_tree = build_SegKDTree(datas[i], labels[i], levels)
        split_dim_v = [np.array(item).astype(np.int64) for item in split_dim]
        split_dims_v.append(split_dim_v)
        points_v.append(point_tree[-1].transpose(0, 2, 1))
        labels_v.append(label_tree[-1].transpose(1, 0))

    split_dims_v = np.array(split_dims_v)
    points_v = np.array(points_v)
    labels = np.array(labels_v)
    print('==========load over==========')

    index_list = list(range(len(datas)))

    def KDSegDataGenerator():
        if mode == 'train':
            random.shuffle(index_list)
        for i in index_list:
            label = np.reshape(labels[i], [-1, 1024, 1]).astype('int64')
            split_dim_v = split_dims_v[i]
            point_v = points_v[i].astype('float32')
            yield [point_v, split_dim_v], label

    return KDSegDataGenerator
