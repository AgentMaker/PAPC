import numpy as np
import random
from PAPC.datasets.tools.build_VoxData import *

def VoxDataLoader(max_point=1024, batchsize=64, mode='train'):
    if mode == 'train':
        file_path = './data/train.txt'
    else:
        file_path = './data/test.txt'

    datas = []
    labels = []
    f = open(file_path)
    for data_list in f:
        point_data = np.load(data_list.split(' ')[0])
        datas.append(point_data)
        labels.append(category[data_list.split(' ')[1].split('\n')[0]])
    f.close()
    datas = np.array(datas)
    labels = np.array(labels)

    index_list = list(range(len(datas)))

    def VoxDataGenerator():
        if mode == 'train':
            random.shuffle(index_list)
        datas_list = []
        labels_list = []
        for i in index_list:
            data = np.reshape(datas[i], [1, 32, 32, 32]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
            datas_list.append(data)
            labels_list.append(label)
            if len(datas_list) == batchsize:
                yield np.array(datas_list), np.array(labels_list)
                datas_list = []
                labels_list = []
        if len(datas_list) > 0:
            yield np.array(datas_list), np.array(labels_list)

    return VoxDataGenerator