import os
import numpy as np
import random
import h5py
from PAPC.datasets.datalist import *

def PNClasDataLoader(max_point=1024, batchsize=64, path='./data/', mode='train'):
    global train_list, test_list, val_list

    datas = []
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
    labels = np.array(labels)
    print('==========load over==========')

    index_list = list(range(len(datas)))

    def PNClasDataGenerator():
        if mode == 'train':
            random.shuffle(index_list)
        datas_list = []
        labels_list = []
        for i in index_list:
            datas_list.append(datas[i].T.astype('float32'))
            labels_list.append(labels[i].astype('int64'))
            if len(datas_list) == batchsize:
                yield np.array(datas_list), np.array(labels_list)
                datas_list = []
                labels_list = []
        if len(datas_list) > 0:
            yield np.array(datas_list), np.array(labels_list)

    return PNClasDataGenerator

def PNSegDataLoader(max_point=1024, batchsize=64, path='./data/', mode='train'):
    global train_list, test_list, val_list
    datas = []
    labels = []
    targets = []
    if mode == 'train':
        for file_list in train_list:
            f = h5py.File(os.path.join(path, file_list), 'r')
            datas.extend(f['data'][:, :max_point, :])
            labels.extend(f['label'])
            targets.extend(f['pid'][:, :max_point])
            f.close()
    elif mode == 'test':
        for file_list in test_list:
            f = h5py.File(os.path.join(path, file_list), 'r')
            datas.extend(f['data'][:, :max_point, :])
            labels.extend(f['label'])
            targets.extend(f['pid'][:, :max_point])
            f.close()
    else:
        for file_list in val_list:
            f = h5py.File(os.path.join(path, file_list), 'r')
            datas.extend(f['data'][:, :max_point, :])
            labels.extend(f['label'])
            targets.extend(f['pid'][:, :max_point])
            f.close()

    datas = np.array(datas)
    labels = np.array(labels)
    targets = np.array(targets)
    print('==========load over==========')

    index_list = list(range(len(datas)))

    def PNSegDataGenerator():
        if mode == 'train':
            random.shuffle(index_list)
        datas_list = []
        labels_list = []
        targets_list = []
        for i in index_list:
            target = np.reshape(targets[i], [max_point, -1]).astype('int64')
            datas_list.append(datas[i].T.astype('float32')) 
            labels_list.append(labels[i].astype('int64'))
            targets_list.append(target)
            if len(datas_list) == batchsize:
                yield [np.array(datas_list), np.array(labels_list)], np.array(targets_list)
                datas_list = []
                labels_list = []
                targets_list = []
        if len(datas_list) > 0:
            yield [np.array(datas_list), np.array(labels_list)], np.array(targets_list)

    return PNSegDataGenerator