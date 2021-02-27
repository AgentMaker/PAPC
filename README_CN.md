# PAPC

PAPC是使用飞桨(PaddlePaddle)框架搭建的用于深度学习的点云处理套件。它的名字PAPC是取Paddle中的PA和PointCloud中的PC组合而来。

## 模型库
### 分类模型
- [VoxNet](./PAPC/models/classify/voxnet)
- [Kd-Networks](./PAPC/models/classify/kdnet)
- [PointNet-Basic](./PAPC/models/classify/pointnet_base)
- [PointNet](./PAPC/models/classify/pointnet)
- [PointNet++SSG](./PAPC/models/classify/pointnet2)
- [PointNet++MSG](./PAPC/models/classify/pointnet2)
### 分割模型
- [Kd-Unet](./PAPC/models/segment/kdunet)
- [PointNet-Basic](./PAPC/models/segment/pointnet_base)
- [PointNet](./PAPC/models/segment/pointnet)
- [PointNet++SSG](./PAPC/models/segment/pointnet2)
- [PointNet++MSG](./PAPC/models/segment/pointnet2)

## 数据集
在ShapeNet数据集(.h5格式)上训练。支持自定义数据集(数据格式参照ShapeNet数据集)。

## 安装

### 1. 安装PaddlePaddle

版本要求

* PaddlePaddle >= 2.0.0rc

* Python >= 3.6+

由于图像分割模型计算开销大，推荐在GPU版本的PaddlePaddle下使用PaddleSeg。推荐安装10.0以上的CUDA环境。安装教程请见[PaddlePaddle官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/install/index_cn.html)。

### 2. 下载PAPC仓库

```shell
git clone https://github.com/AgentMaker/PAPC.git
```


## 训练

### 训练参数查看
```shell
python train.py --help
```

### 快速训练
```shell
python train.py
```

## 更好的体验
我们的项目挂载在百度AIStudio上面，AIStudio能够提供免费的GPU资源用于训练，类似Google的Colab一样。您可以在AIStudio上面自由地运行这个项目且是免费的。<br><br>
项目网址1: [Origin PAPC Project](https://aistudio.baidu.com/aistudio/projectdetail/1531789)<br>
项目网址2: [PAPC Project](https://aistudio.baidu.com/aistudio/projectdetail/1555858)


## 联系我们
Email : [agentmaker@163.com]()
