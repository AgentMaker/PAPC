import argparse
from PAPC.train import train

parser = argparse.ArgumentParser(description='PAPC Initialization')
parser.add_argument('--model_name', type=str, default='pointnet_basic', help='The name of model, such as pointnet, pointnet2 and so on')
parser.add_argument('--mode', type=str, default='clas', help='"clas", "seg" or "detect"')
parser.add_argument('--max_point', type=int, default=1024, help='How many points in a sample during training')
parser.add_argument('--num_classes', type=int, default=16, help='How many classes in classification during training')
parser.add_argument('--num_parts', type=int, default=50, help='How many classes in segmentation during training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
parser.add_argument('--epoch_num', type=int, default=10, help='epoch for training')
parser.add_argument('--batchsize', type=int, default=32, help='Mini batch size of one gpu or cpu')
parser.add_argument('--info_iter', type=int, default=40, help='How many iters to info measurement during training')
parser.add_argument('--save_iter', type=int, default=2, help='How many iters to save a model snapshot once during training')
parser.add_argument('--path', type=str, default='./dataset/', help='The directory for finding dataset')

args = parser.parse_args()

if __name__ == '__main__':
    train(args.model_name,
        args.mode,
        args.max_point,
        args.num_classes,
        args.num_parts,
        args.learning_rate,
        args.weight_decay,
        args.epoch_num,
        args.batchsize,
        args.info_iter,
        args.save_iter,
        args.path)