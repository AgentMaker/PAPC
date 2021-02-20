from PAPC.datasets.kdloader import KDClasDataLoader, KDSegDataLoader
from PAPC.datasets.voxloader import VoxDataLoader
from PAPC.datasets.pnloader import PNClasDataLoader, PNSegDataLoader

def DataLoader(model_name, max_point, batchsize, path='./data/', mode1='clas' ,mode2='train'):
    if mode1 == 'clas':
        if model_name == 'voxnet':
            return VoxDataLoader(max_point, batchsize, mode2)
        elif model_name == 'kdnet':
            return KDClasDataLoader(max_point, batchsize, path, mode2)
        elif model_name == 'pointnet_basic':
            return PNClasDataLoader(max_point, batchsize, path, mode2)
        elif model_name == 'vfe':
            return PNClasDataLoader(max_point, batchsize, path, mode2)
        elif model_name == 'pointnet2_ssg':
            return PNClasDataLoader(max_point, batchsize, path, mode2)
        elif model_name == 'pointnet2_msg':
            return PNClasDataLoader(max_point, batchsize, path, mode2)
        else:
            raise SystemExit('Error: model is incorrect')
    elif mode1 == 'seg':
        if model_name == 'kdunet':
            return KDSegDataLoader(max_point, batchsize, path, mode2)
        elif model_name == 'pointnet_basic':
            return PNSegDataLoader(max_point, batchsize, path, mode2)
        elif model_name == 'vfe':
            return PNSegDataLoader(max_point, batchsize, path, mode2)
        elif model_name == 'pointnet2_ssg':
            return PNSegDataLoader(max_point, batchsize, path, mode2)
        elif model_name == 'pointnet2_msg':
            return PNSegDataLoader(max_point, batchsize, path, mode2)
        else:
            raise SystemExit('Error: model is incorrect')
    elif mode1 == 'detect':
        raise SystemExit('Error: Sorry, do not have detect model')
    else:
        raise SystemExit('Error: mode should be "clas", "detect" or "seg"')