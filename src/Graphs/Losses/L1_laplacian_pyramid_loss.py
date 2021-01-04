import numpy as np
import torch
import torch.nn.functional as F

from .base_convergent_loss import Base_Convergent_Loss


class L1_Laplacian_Pyramid_Loss(Base_Convergent_Loss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_channels = self.get_cfgs('num_channels')
        self.pyramid_levels = self.get_cfgs('pyramid_levels')
        self.kernel_size = self.get_cfgs('kernel_size')
        self.sigma = self.get_cfgs('sigma')
        self.kernel = self.init_kernel()
        self.coef = self.get_cfgs('loss_coef', default=.1)

    def init_kernel(self):
        assert(self.kernel_size % 2 == 1),\
            'Kernel size should be odd but it is %d' % (self.kernel_size)
        grid = np.float32(np.mgrid[0:self.kernel_size, 0:self.kernel_size].T)

        def gaussian(x):
            return np.exp((x - self.kernel_size // 2)**2 / (-2 * self.sigma**2))**2

        self.kernel = np.sum(gaussian(grid), axis=2)
        self.kernel /= np.sum(self.kernel)
        self.kernel = np.tile(self.kernel, (self.num_channels, 1, 1))
        self.kernel = self.modality.wrap(self.kernel[:, None, :, :])
        return self.kernel

    def laplacian_pyramid(self, img):
        current = img
        pyramid = []

        for _ in range(self.pyramid_levels):
            filtered = self.conv_gauss(current)
            diff = current - filtered
            pyramid.append(diff)
            current = F.avg_pool2d(filtered, 2)

        pyramid.append(current)
        return pyramid

    def conv_gauss(self, img):
        pad = tuple([self.kernel_size // 2] * 4)
        img = F.pad(img, pad, mode='replicate')
        return F.conv2d(img, self.kernel, groups=self.num_channels)

    def calculate_loss(self, output, target):
        pyramid_output = self.laplacian_pyramid(torch.tanh(output))
        pyramid_target = self.laplacian_pyramid(target.detach())
        return sum(F.l1_loss(o, t) for o, t in zip(pyramid_output, pyramid_target))
