import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):   #size=300,base=vgg,head = loc,conf
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes           #类别
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)            #box数目
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)        #vgg网络的使用
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)        #Conv4_3为38*38，正则化的使用
        self.extras = nn.ModuleList(extras)      #额外的网络架构

        self.loc = nn.ModuleList(head[0])             #生成位置逻辑回归的网络
        self.conf = nn.ModuleList(head[1])               #生成类别置信度的网络

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)           #调用detection.py

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()            # source用于检测的feature map, 这些分支后要接loc + conf模块，用于bbox预测

        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):                  #第一个特征图，来自vgg的conv4_3
            x = self.vgg[k](x)

        s = self.L2Norm(x)     #调用l2norm.py
        sources.append(s)                  #第一个特征图经过norm，填充到source中

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):    
            x = self.vgg[k](x)
        sources.append(x)                   #conv7（原fc7）填充到source中

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):      #extra layers四个特征图用于检测
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        # 如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
        # source：需要做bbox预测的分支
        #self.loc：做bbox offsets预测的分支
        #self.conf：做bbox cls预测的分支
        # permute是维度换位，可以灵活的对原数据的维度进行调换，而数据本身不变。permute(2,3,1,4)，按照2,3,1,4维进行重排列。
        #（batch_size,c,h,w) ---->(batch_size,h,w,c)
        for (x, l, c) in zip(sources, self.loc, self.conf):        
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        # view(out.size(0), -1), 目的是将多维的的数据如（none，36，2，2）平铺为一维如（none，144）
        # 相当于concate了所有检测分支上的检测结果
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:    #训练的时候
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):                 #下方调用时传入cfg=base['300'],i=3
    layers = []                                            # 用于存放vgg网络的list
    in_channels = i                                        # 默认RGB图像，因此通道数一般为3
    for v in cfg:                                          # 多层循环，数据信息存放在字典vgg_base中
        if v == 'M':                                         # maxpooling，
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':                                        # maxpooling，边缘补NAN
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:                                                 # 卷积前后维度读取字典中数据
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:                                       #BN层
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:                                             #ReLU层
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)   # 空洞卷积，扩张率为6（可以扩大卷积感受野的范围，但没有增加卷积size）
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

# 定义新添加的特征层
def add_extras(cfg, i, batch_norm=False):                      #下方调用时传入cfg=extras['300'],i=1024
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):                            # 多层循环，数据信息存放在字典extras_base中
        if in_channels != 'S':                            # S代表stride，为2时候就相当于缩小feature map
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):    #cfg对应了下方mbox['300']。（可从下方程序看出）
    loc_layers = []                      #输出维度是default box的种类(4or6)*4
    conf_layers = []                     #输出维度是default box的种类(4or6)*num_class
                                         # 预测分支是全卷积的，4对应bbox坐标，num_classes对应预测目标类别，如VOC = 21
    vgg_source = [21, -2]                 # 第21层和倒数第二层
    #选取的特征图在输入的两个list(vgg和extra_layer)中的索引：
    #vgg:21,-2分别对应conv4_3去掉relu的末端；conv7relu之前的1*1卷积
    #extra_layer:1,3,5,7;对应conv8_2,conv9_2,conv10_2,conv11_2
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    # 对应的参与检测的分支数
    for k, v in enumerate(extra_layers[1::2], 2):              # 找到对应的层    
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):           # 构建SSD的接口函数
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    #融合到multibox中给出对应的vgg,extral_layer,还有生成处理置信度和位置回归的网络，最后放入SSD整体网络中
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
