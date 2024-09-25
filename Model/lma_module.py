from torch import nn
import math

__all__ = ['LMALayer']


class LMALayer(nn.Module):
    """Constructs a LMA module.
    Args:
        inp: Number of channels of the input feature maps
    """
    def __init__(self, inp):
        super(LMALayer, self).__init__()
        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        k = temp if temp % 2 else temp - 1

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_h = nn.Conv2d(1, 1, kernel_size=(1, k), stride=1, padding=(0, (k - 1) // 2), bias=False)
        self.conv_w = nn.Conv2d(1, 1, kernel_size=(1, k), stride=1, padding=(0, (k - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x).permute(0, 3, 2, 1).contiguous()
        x_w = self.pool_w(x).permute(0, 2, 3, 1).contiguous()

        x_h = self.conv_h(x_h).permute(0, 3, 2, 1).contiguous()
        x_w = self.conv_w(x_w).permute(0, 3, 1, 2).contiguous()

        x_h = self.sigmoid(x_h)
        x_w = self.sigmoid(x_w)

        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y
