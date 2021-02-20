import os
import numpy as np

category = {
    'bathtub': 0,
    'bed': 1,
    'chair': 2,
    'door': 3,
    'dresser': 4,
    'airplane': 5,
    'piano': 6,
    'sofa': 7,
    'person': 8,
    'cup': 9
}
_category = {
    0: 'bathtub',
    1: 'bed',
    2: 'chair',
    3: 'door',
    4: 'dresser',
    5: 'airplane',
    6: 'piano',
    7: 'sofa',
    8: 'person',
    9: 'cup'
}
categoryList = [
    'bathtub',
    'bed',
    'chair',
    'door',
    'dresser',
    'airplane',
    'piano',
    'sofa',
    'person',
    'cup'
]

def transform():
    for i in range(len(categoryList)):
        dirpath = os.path.join('./modelnet40_normal_resampled', categoryList[i])
        dirlist = os.listdir(dirpath)
        if not os.path.exists(os.path.join('./data', categoryList[i])):
            os.mkdir(os.path.join('./data', categoryList[i]))
        for datalist in dirlist:
            datapath = os.path.join(dirpath, datalist)
            zdata = []
            xdata = []
            ydata = []
            f = open(datapath, 'r')
            for point in f:
                xdata.append(float(point.split(',')[0]))
                ydata.append(float(point.split(',')[1]))
                zdata.append(float(point.split(',')[2]))
            f.close()
            arr = np.zeros((32,) * 3).astype('float32')
            for j in range(len(xdata)):
                arr[int(xdata[j] * 15.5 + 15.5)][int(ydata[j] * 15.5 + 15.5)][int(zdata[j] * 15.5 + 15.5)] = 1
            savepath = os.path.join('./data', categoryList[i], datalist.split('.')[0]+'.npy')
            np.save(savepath, arr)

def getDatalist():
    f_train = open('./data/train.txt', 'w')
    f_test = open('./data/test.txt', 'w')
    for i in range(len(categoryList)):
        dict_path = os.path.join('./data/', categoryList[i])
        data_dict = os.listdir(dict_path)
        count = 0
        for data_path in data_dict:
            if count % 60 != 0:
                f_train.write(os.path.join(dict_path, data_path) + ' ' + categoryList[i]+ '\n')
            else:
                f_test.write(os.path.join(dict_path, data_path) + ' ' + categoryList[i]+ '\n')
            count += 1
    f_train.close()
    f_test.close()


if __name__ == '__main__':
    transform()
    getDatalist()