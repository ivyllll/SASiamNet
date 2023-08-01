import os,argparse
from random import random

import numpy as np
import torch
from torch import seed

from LoadData import Change_Detect
from train import train
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from prettytable import PrettyTable
import warnings
warnings.filterwarnings('ignore')

# from UNet import UNet as MY_NET
# from FC_CONC import SiamUnet_conc as MY_NET
# from FC_EF import FC_EF as MY_NET
# from FC_DIFF import FC_Siam_diff as MY_NET
# from ResNet import resnet18 as MY_NET
# from MFGANnet import MFGANnet as MY_NET
# from unet_2 import UNet as MY_NET
# from xxNet_v5 import BaseNet as MY_NET
# from xxNet_v11_noPFFM import BaseNet as MY_NET
from xxNet_v11_noHIRM import BaseNet as MY_NET
# from TemporalFeatureInteractionModule import BaseNet as MY_NET

def main(params):
    # seed_everything()
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs',          type=int,       default=5,)
    parser.add_argument('--checkpoint_step',     type=int,       default=5, )
    parser.add_argument('--validation_step',     type=int,       default=1, )
    parser.add_argument('--batch_size',          type=int,       default=4, )
    parser.add_argument('--num_workers',         type=int,       default=4, )
    parser.add_argument('--lr',                  type=float,     default=0.0001, )
    parser.add_argument('--lr_scheduler',        type=int,       default=3, )
    parser.add_argument('--lr_scheduler_gamma',  type=float,     default=0.99,       help='learning rate attenuation coefficient')
    parser.add_argument('--warmup',              type=int,       default=1,          help='warm up')
    parser.add_argument('--cuda',                type=str,       default='0')
    parser.add_argument('--DataParallel',        type=int,       default=1,          help='train in multi GPU')
    parser.add_argument('--beta1',               type=float,     default=0.5)
    parser.add_argument('--beta2',               type=float,     default=0.999)
    parser.add_argument('--miou_max',            type=float,     default=0.85)
    parser.add_argument('--pretrained_model_path',  type=str,       default=None,       help='None')
    parser.add_argument('--save_model_path',     type=str,       default="./checkpoint")
    parser.add_argument('--data',                type=str,       default="D:/.ipynb_checkpoints/CD/DATA/CDD")
    parser.add_argument('--log_path',            type=str,       default="./log")
    parser.add_argument('--result',              type=str,       default="./")
    parser.add_argument('--model_name',          type=str,       default='dfn',      help='')
    parser.add_argument('--dir_name',            type=str,       default='dfn',     help='')


    args=parser.parse_args(params)

    # 打印 params 的信息
    tb = PrettyTable(['Num', 'Key', 'Value'])
    args_str = str(args)[10:-1].split(', ')
    for i, key_value in enumerate(args_str):
        key, value = key_value.split('=')[0], key_value.split('=')[1]
        tb.add_row([i + 1, key, value])
    print(tb)

    # 检查文件夹是否存在，不存在就创建
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    # if not os.path.exists(args.result):
    #     os.makedirs(args.result)

    if not os.path.exists(f'{args.save_model_path}/{args.model_name}'):
        os.makedirs(f'{args.save_model_path}/{args.model_name}')
    # if not os.path.exists(args.summary_path+args.dir_name+'/checkpoints'):
    #     os.makedirs(args.summary_path+args.dir_name+'/checkpoints')

    # 创建数据集和数据加载器
    # 训练集
    train_path_img1  = os.path.join(args.data, 'train/img1')
    train_path_img2  = os.path.join(args.data, 'train/img2')
    train_path_label = os.path.join(args.data, 'train/label')# os.path.join()函数：连接两个或更多的路径名组件
    # 验证集
    val_path_img1    = os.path.join(args.data, 'test/img1')
    val_path_img2    = os.path.join(args.data, 'test/img2')
    val_path_label   = os.path.join(args.data, 'test/label')
    #
    csv_path         = os.path.join(args.data, 'class_dict.csv')


    # 训练集数据加载
    dataset_train    = Change_Detect(
                                    train_path_img1,
                                    train_path_img2,
                                    train_path_label,
                                    csv_path,
                                    mode='train',
                                    )
    dataloader_train = DataLoader(
                                    dataset_train,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    # drop_last=True,
                                    num_workers=args.num_workers,
                                  )
    # 验证集数据加载
    dataset_val      =Change_Detect(
                                    val_path_img1,
                                    val_path_img2,
                                    val_path_label,
                                    csv_path,
                                    mode='val'
                                    )
    dataloader_val = DataLoader(
                                    dataset_val,
                                    batch_size=1,  # must 1
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                )

    #
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda #s.environ[‘环境变量名称’]=‘环境变量值’
    torch.backends.cudnn.benchmark = True
    # 选择模型

    # model = MY_NET() # FC_DIFF FC_CONC
    # model = MY_NET(2)
    model = MY_NET(6,2)   # FC_EF


    # 设置GPU
    # if args.DataParallel == 1:
    #     print('mulit Cuda! cuda:{:}'.format(args.cuda))
    #     model = torch.nn.DataParallel(model)
    #     model = model.cuda()
    # else:
    #     print('Single cuda!')
    #     model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Adam梯度下降
    optimizer = torch.optim.Adam(model.parameters(),args.lr,(args.beta1,args.beta1))
    lr_scheduler = StepLR(optimizer,step_size=args.lr_scheduler,gamma=args.lr_scheduler_gamma)

    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)

        # loading the part of network params
        pretrained_dict = torch.load(args.pretrained_model_path)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        print('Done!')

    # 开始训练
    train(args, model, optimizer, dataloader_train, dataloader_val, lr_scheduler)


# def seed_everything():
#     '''
#     设置整个开发环境的seed
#     :param seed:
#     :param device:
#     :return:
#     '''
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#     # some cudnn methods can be random even after fixing the seed
#     # unless you tell it to be deterministic
#     torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    params = [
        '--num_epochs', '200',
        '--batch_size', '2',
        '--lr', '0.0015',
        '--warmup', '0',
        '--lr_scheduler_gamma', '0.9',
        '--lr_scheduler', '4',
        '--miou_max', '0.50',
        '--DataParallel', '1',  # 1: True  0:False
        '--cuda', '0',  # model put in the cuda[0]
        '--checkpoint_step', '10',
        '--result', 'FC_CONC(CDD batch2)',
        '--model_name','FC_CONC(CDD batch2)',
        # '--result', 'MFGANnet',
        # '--model_name','MFGANnet',
        # '--dir_name', 'FC_DIFF',
        # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
            ]

    main(params)
