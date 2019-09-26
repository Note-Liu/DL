from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']                       ## 输入的图像尺度
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])             # 各个feature map上预定义的anchor长宽比清单，与检测分支的数量对应
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']               # 特征金字塔层上各个feature map尺度
        self.min_sizes = cfg['min_sizes']                     # 预定义的anchor尺度
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']     # feature map上每个pix上预定义的anchor,[2, 3]对应6个anchor,[2]对应4个anchor
        self.clip = cfg['clip']                        # 位置校验
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        mean = []                                  # 保存所有feature map上预定义的anchor
        for k, f in enumerate(self.feature_maps):    # 对特征金字塔的各个检测分支，每个feature map上each-pixel都做密集anchor采样
            # 笛卡尔乘积(product(A, B)函数，返回A、B中的元素的笛卡尔积的元组)
            # product(list1, list2) 依次取出list1中的每1个元素，与list2中的每1个元素，组成元组，然后将所有的元组组成一个列表，返回。
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y     矩形框的中心点坐标
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                # small sized square box，小尺寸的方框
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                # big sized square box，大尺寸的方框
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # change h/w ratio of the small sized box，2/4个长方形框
                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # 总结：
        # feature map上each-pixel对应4 / 6个anchor，长宽比：2:1 + 1:2 + 1:3 + 3:1 + 1:1 + 1:1，后两个1:1的anchor对应的尺度有差异；
        # 所有feature map上所有预定义的不同尺度、长宽比的anchor保存至mean中。

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)         # float型坐标校验
        return output
