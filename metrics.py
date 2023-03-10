# -*- coding: utf-8 -*-
import torch
import numpy as np

def psnr(im0, im1, eps=10e-8):
    """ This function computes the Peak Signal to Noise Ratio (PSNR) between two images whose ranges are [0-1].
        Args:
            im0 (torch.tensor): Image 0, should be of same shape and type as im1
            im1 (torch.tensor): Image 1,  should be of same shape and type as im0
        Returns:
            torch.tensor (): Returns the mean PSNR value for the complete image.
        """
    if torch.is_tensor(im0):
        # im0 = im0.cpu().detach().numpy()
        # im0 = im1.cpu().detach().numpy()
        out = -10.0*torch.log10(torch.mean(torch.pow(im0-im1, 2.0)+eps))
        out = out.detach().cpu().numpy()
        return out
    else :
        return -10*np.log10(np.mean(np.power(im0-im1, 2))+eps)


class MetricAll():
    def __init__(self):
        """
        (B,3H,W)
        :param opt:
        """
        # assert opt!=None,"options is required"
        super(MetricAll, self).__init__()
        self.psnr_list = np.array([])
        self.psnr_statics = {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "var": 0.0
        }

    def compute_metrics(self,gt, out):
        for i in range(len(gt)):
            self.psnr_list = np.append(
                self.psnr_list,
                psnr(gt[i,:,:,:], out[i,:,:,:])
            )
        return None

    def compute_statics(self):
        self.psnr_statics = {
            "min": self.psnr_list.min(),
            "max": self.psnr_list.max(),
            "mean":self.psnr_list.mean(),
            "var": self.psnr_list.var()
        }
        return None

    def clear_all(self):
        self.psnr_list = np.array([])

        self.psnr_statics = {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "var": 0.0
        }
        return None