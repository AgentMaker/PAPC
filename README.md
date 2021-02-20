English | [简体中文](README_CN.md)

# PAPC

Welcome to PAPC(Paddle PointCloud) which is a deep learning for point clouds platform based on pure PaddlePaddle.


## Model Zoo
### Clas
- [VoxNet](./PAPC/models/classify/voxnet.py)
- [Kd-Networks](./PAPC/models/classify/kdnet.py)
- [PointNet-Basic](./PAPC/models/classify/pointnet_base.py)
- [PointNet](./PAPC/models/classify/pointnet.py)
- [PointNet++SSG](./PAPC/models/classify/pointnet2.py)
- [PointNet++MSG](./PAPC/models/classify/pointnet2.py)
### Seg
- [Kd-Unet](./PAPC/models/segment/kdunet.py)
- [PointNet-Basic](./PAPC/models/segment/pointnet_base.py)
- [PointNet](./PAPC/models/segment/pointnet.py)
- [PointNet++SSG](./PAPC/models/segment/pointnet2.py)
- [PointNet++MSG](./PAPC/models/segment/pointnet2.py)


## Installation

### step 1. Install PaddlePaddle

System Requirements:
* PaddlePaddle >= 2.0.0rc
* Python >= 3.6+

Highly recommend you install the GPU version of PaddlePaddle, due to large overhead of segmentation models, otherwise it could be out of memory while running the models. For more detailed installation tutorials, please refer to the official website of [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-beta/install/index_cn.html)。

### step 2. Download PAPC repo

```shell
git clone https://github.com/AgentMaker/PAPC.git
```


## Help For Training
```shell
python train.py --help
```


## Quick Training
```shell
python train.py
```


## Contact us
Email : [agentmaker@163.com]()