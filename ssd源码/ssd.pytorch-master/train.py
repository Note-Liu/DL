from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys                                      #sys模块包含了于python解释器和它的环境有关的函数
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn  
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,                       #在voc0712.py中，VOC数据集路径
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',                #主干网络预训练权重
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,                       #batch_size
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')   #恢复训练
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,                       #多线程读取数据的线程数
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,          #损失可视化
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():                                       #硬件cuda
    if args.cuda:                                                   #软件设置
        torch.set_default_tensor_type('torch.cuda.FloatTensor')      #默认FloatTensor型
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    #选择数据集
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],     #在config.py中coco['min_dim']=300大小
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc                                                              # SSD300 CONFIGS
        dataset = VOCDetection(root=args.dataset_root,                         #调用voc0712.py
                               transform=SSDAugmentation(cfg['min_dim'],       #在config.py中voc['min_dim']=300大小
                                                         MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])   #在ssd.py中，构建ssd网络
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)    #如果有多个GPU，多GPU并行
        cudnn.benchmark = True

    # if torch.cuda.device_count() > 1:　　　＃补充，多GPU的语句
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model)
    # if torch.cuda.is_available():
    #     model.cuda()

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)
　　
    if args.cuda:
        net = net.cuda()
    
    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,        #调用multibox_loss.py
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size             #训练一遍整个数据集需要多少个epoch
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        #在forward之前完成修改lr
        if iteration in cfg['lr_steps']:          #调用config.py   
            step_index += 1                        #'lr_steps': (80000, 100000, 120000)，当iteration为这三个值时，index+1
            adjust_learning_rate(optimizer, args.gamma, step_index)  #调用此函数需要输入optimizer等参数，并且args.lr作为初始化的学习率也需要给出

        # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]  #volatile=True,不会被求导
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]

        if iteration % 10 == 0:              # 每10次迭代输出一次
            print('timer: %.4f sec.' % (t1 - t0))                
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')

        if args.visdom:
            update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:       #每迭代5000次，保存一次训练权重
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(),                                #训练结束保存模型
               args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):          #调用此函数需要传入优化器,  该函数只被调用三次
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))        #default gamma = 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr                  #lr改为当前新的lr
#lr = args.lr * (0.1 ** (epoch // 30))   Sets the learning rate to the initial LR decayed by 10 every 30 epochs

def xavier(param):
    init.xavier_uniform(param)   #xavier参数初始化，均匀分布


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
