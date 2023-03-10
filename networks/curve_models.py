import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.dual_attention import TripleAttention

class ChebyshevPol(nn.Module):
    """
    The Chebyshev polynomials of the first kind are obtained from the recurrence relation：
    I_0 = 1
    I_1 = x
    I_{n+1} = 2 x I_n - I_{n-1}
    """
    def __init__(self):
        super(ChebyshevPol, self).__init__()

    def forward(self, x, xk, xk1):
        return 2 * x * xk - xk1


class ChebySumBlock3DAB(nn.Module):
    """
    sumation
    """
    def __init__(self, num_img_in=3, num_feature=96, adb_type="baseline"):
        super(ChebySumBlock3DAB, self).__init__()
        # self.fcat = ResidualFusion3()
        if adb_type=="triple":
            self.est_a = TripleAttention(channel=3, num_feature=num_feature, kernel_size=3)
        # ablation study
        # self.est_a = AttentionCNN(in_channels = 3, num_feature=64, out_channels = 3)

    def forward(self, x, y, xk):
        # tmp = self.fcat()
        a = self.est_a(x, y, xk)
        return a * xk + y, a

class ChebyAll3DAB(nn.Module):
    """
     cat x,y,Tn
    """
    def __init__(self, num_blocks=6, require_a=False,num_feature=96,adb_type="baseline"):
        super(ChebyAll3DAB, self).__init__()
        self.require_a = require_a
        self.num_blocks = num_blocks
        self.abp1 = ChebySumBlock3DAB(num_img_in=3, num_feature=num_feature,adb_type=adb_type)
        self.cbpList = nn.ModuleList([ChebyshevPol() for i in range(num_blocks-2)])
        self.abpList = nn.ModuleList([ChebySumBlock3DAB(num_img_in=3, num_feature=num_feature,adb_type=adb_type) for i in range(num_blocks-1)])

    def slow_forward(self, x):
        img = []
        a = []
        y = []
        img1 = torch.ones_like(x)
        img2 = x
        img.append(img1)
        img.append(img2)
        for blk in self.cbpList:
            imgk = blk(x, img[-1], img[-2])
            img.append(imgk)
        y1, a1 = self.abp1(x, x, img[0])
        y.append(y1)
        a.append(a1)
        for i, blk in enumerate(self.abpList):
            yk, ak = blk(x, y[-1], img[i+1])
            y.append(yk)
            a.append(ak)

        if self.require_a:
            return y, a, img#, out
            #return y[-1]
        else:
            out = y[-1]
            return out

    def fast_forward(self, x):
        img1 = torch.ones_like(x)
        img2 = x
        for k in range(self.num_blocks):
            if k==0:
                y, _ = self.abp1(x,x,img1) # layer 1
                img1, img2 = img2, self.cbpList[k](x, img2, img1)
            elif k==self.num_blocks-2: 
                y, _ = self.abpList[k-1](x, y, img1)
                img1 = img2
            elif k==self.num_blocks-1: # the last layer 
                out, _ = self.abpList[k-1](x, y, img1)
            else:
                y, _ = self.abpList[k-1](x,y,img1)
                img1, img2 = img2, self.cbpList[k](x, img2, img1)
        return out

    def forward(self,x):
        if self.require_a:
            return self.slow_forward(x)
        else:
            return self.fast_forward(x)