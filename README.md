English | [简体中文](README_CN.md)

# PAPC

Welcome to PAPC(Paddle PointCloud) which is a deep learning for point clouds platform based on pure PaddlePaddle.

## Show

![](https://ai-studio-static-online.cdn.bcebos.com/c43ee4383fa447f6abd3c6b6f27110d9ea649fa453f64b74af6b9753af3405ad)

## Model Zoo
### Clas
- [VoxNet](./PAPC/models/classify/voxnet)
- [Kd-Networks](./PAPC/models/classify/kdnet)
- [PointNet-Basic](./PAPC/models/classify/pointnet_base)
- [PointNet](./PAPC/models/classify/pointnet)
- [PointNet++SSG](./PAPC/models/classify/pointnet2)
- [PointNet++MSG](./PAPC/models/classify/pointnet2)
### Seg
- [Kd-Unet](./PAPC/models/segment/kdunet)
- [PointNet-Basic](./PAPC/models/segment/pointnet_base)
- [PointNet](./PAPC/models/segment/pointnet)
- [PointNet++SSG](./PAPC/models/segment/pointnet2)
- [PointNet++MSG](./PAPC/models/segment/pointnet2)
### Detect
- [Pointpillars](./PAPC/models/detect/pointpillars)


## Dataset
Based on ShapeNet dataset(.h5 format). Support custom dataset(data format refered to ShapeNet dataset).


## Installation

### Step 1. Install PaddlePaddle

System Requirements:
* PaddlePaddle >= 2.0.0rc
* Python >= 3.6+

Highly recommend you install the GPU version of PaddlePaddle, due to large overhead of pointcloud models, otherwise it could be out of memory while running the models. For more detailed installation tutorials, please refer to the official website of [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-beta/install/index_cn.html)。

### Step 2. Download PAPC repo

```shell
git clone https://github.com/AgentMaker/PAPC.git
```


## Training

### Help For Training
```shell
python train.py --help
```

### Quick Training
```shell
python train.py
```


## Better Experience
This project is mounted on Baidu AIStudio which provides a free GPU environment like Google Colab. You can run this project on it for free. <br><br>
Url1: [Origin PAPC Project](https://aistudio.baidu.com/aistudio/projectdetail/1531789)<br>
Url2: [PAPC Project](https://aistudio.baidu.com/aistudio/projectdetail/1555858)

## Contact us
Email : [agentmaker@163.com]()<br>
QQ Group : 1005109853
