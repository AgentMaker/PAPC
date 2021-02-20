English | [简体中文](README_CN.md)

# PAPC

Welcome to PAPC(Paddle PointCloud) which is a deep learning for point clouds platform based on pure PaddlePaddle.


## Model Zoo

- [VoxNet](./PAPC/classify/voxnet.py)


## Installation

#### step 1. Install PaddlePaddle

System Requirements:
* PaddlePaddle >= 2.0.0rc
* Python >= 3.6+

Highly recommend you install the GPU version of PaddlePaddle, due to large overhead of segmentation models, otherwise it could be out of memory while running the models. For more detailed installation tutorials, please refer to the official website of [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-beta/install/index_cn.html)。

#### step 2. Download PAPC repo

```shell
git clone https://github.com/AgentMaker/PAPC.git
```


## Help
```shell
python train.py --help
```


## Quick Training
```shell
python train.py
```


## Contact us
Email: [agentmaker@163.com]()