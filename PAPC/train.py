import paddle
from PAPC.models.classify import *
from PAPC.models.detect import *
from PAPC.models.segment import *
from PAPC.models.loss import *
from PAPC.datasets import *

def init_model(model_name='pointnet_basic', mode='clas', num_classes=16, num_parts=50):
    model, loss_fn = None, None
    if mode == 'clas':
        if model_name == 'voxnet':
            model = VoxNet(num_classes=num_classes)
            loss_fn = CrossEntropyLoss()
        elif model_name == 'kdnet':
            model = KDNet(num_classes=num_classes)
            loss_fn = CrossEntropyLoss()
        elif model_name == 'pointnet_basic':
            model = PointNet_Basic_Clas(num_classes=num_classes)
            loss_fn = CrossEntropyLoss()
        elif model_name == 'vfe':
            model = VFE_Clas(num_classes=num_classes)
            loss_fn = CrossEntropyLoss()
        elif model_name == 'pointnet2_ssg':
            model = PointNet2_SSG_Clas(num_classes=num_classes)
            loss_fn = CrossEntropyLoss()
        elif model_name == 'pointnet2_msg':
            model = PointNet2_MSG_Clas(num_classes=num_classes)
            loss_fn = CrossEntropyLoss()
        else:
            raise SystemExit('Error: model is incorrect')
    elif mode == 'seg':
        if model_name == 'kdunet':
            model = KDUNet(num_classes=num_parts)
            loss_fn = CrossEntropyLoss()
        elif model_name == 'pointnet_basic':
            model = PointNet_Basic_Seg(num_classes=num_parts)
            loss_fn = CrossEntropyLoss()
        elif model_name == 'vfe':
            model = VFE_Seg(num_classes=num_parts)
            loss_fn = CrossEntropyLoss()
        elif model_name == 'pointnet2_ssg':
            model = PointNet2_SSG_Seg(num_classes=num_classes, num_parts=num_parts)
            loss_fn = CrossEntropyLoss()
        elif model_name == 'pointnet2_msg':
            model = PointNet2_MSG_Seg(num_classes=num_classes, num_parts=num_parts)
            loss_fn = CrossEntropyLoss()
        else:
            raise SystemExit('Error: model is incorrect')
    elif mode == 'detect':
        raise SystemExit('Error: Sorry, do not have detect model')
    else:
        raise SystemExit('Error: mode should be "clas", "detect" or "seg"')

    return model, loss_fn

def init_optim(model, learning_rate=0.001, weight_decay=0.001):
    optim = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=learning_rate, weight_decay=weight_decay)

    return optim

def info(epoch, batch_id, predicts, targets, loss, mode, num_parts=50):
    if mode == 'clas':
        acc = paddle.metric.accuracy(predicts, targets)
        print("epoch: {}, batch_id: {}, loss is: {}, accuracy is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
    elif mode == 'seg':
        miou, _, _ = paddle.fluid.layers.mean_iou(
                        paddle.unsqueeze(paddle.argmax(predicts, axis=-1), axis=-1), targets, num_parts)
        print("epoch: {}, batch_id: {}, loss is: {}, miou is: {}".format(epoch, batch_id, loss.numpy(), miou.numpy()))
    else:
        pass

def train(
        model_name = 'pointnet_basic',
        mode = 'clas',
        max_point = 1024,
        num_classes = 16,
        num_parts = 50,
        learning_rate = 0.001,
        weight_decay = 0.001,
        epoch_num = 10,
        batchsize = 32,
        info_iter = 40,
        save_iter = 2,
        path = './dataset/'
    ):
    train_loader = DataLoader(model_name, max_point, batchsize, path, mode, 'train')
    val_loader = DataLoader(model_name, max_point, batchsize, path, mode, 'val')

    model, loss_fn = init_model(model_name, mode, num_classes, num_parts)
    model.train()
    optim = init_optim(model, learning_rate, weight_decay)

    for epoch in range(epoch_num):
        # train
        print("===================================train===========================================")
        for batch_id, data in enumerate(train_loader()):
            inputs = data[0]
            targets = paddle.to_tensor(data[1])

            predicts = model(inputs)
            # print(predicts.shape)
            # print(targets.shape)
            loss = loss_fn(predicts, targets)

            if batch_id % info_iter == 0:
                info(epoch, batch_id, predicts, targets, loss, mode, num_parts)

            loss.backward()
            optim.step()
            optim.clear_grad()

        if epoch % save_iter == 0:
            paddle.save(model.state_dict(), './model/'+model_name+'_'+str(epoch)+'.pdparams')
            paddle.save(optim.state_dict(), './model/'+model_name+'_'+str(epoch)+'.pdopt')

        # validation
        print("===================================val===========================================")
        model.eval()
        for batch_id, data in enumerate(val_loader()):
            inputs = data[0]
            targets = paddle.to_tensor(data[1])

            predicts = model(inputs)
            loss = loss_fn(predicts, targets)

            if batch_id % info_iter == 0:
                info(epoch, batch_id, predicts, targets, loss, mode, num_parts)

        model.train()


if __name__ == '__main__':
    train()