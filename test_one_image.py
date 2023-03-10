# -*- coding: utf-8 -*-
# files in the project
import networks.curve_models as mynet
# pytorch
import torch
import torch.optim
import torch.nn.functional as F
# torchvision
from torchvision import transforms
from torchvision.utils import save_image
# utils packages
import numpy as np
from torchsummaryX import summary
import matplotlib
matplotlib.use('agg')
from PIL import Image
# other files in the project
import torch
# utils packages
import os
import argparse
parser = argparse.ArgumentParser()

# test settings
parser.add_argument('--dir_pth', type=str, default="./cheby_LOLv1.pth", help='pretrained model dir')
parser.add_argument("--gpu_id", type=str, default="0", help="ids of gpu to be used")
parser.add_argument('--img_path', type=str, default="/home/pjw/Datasets/LOL/eval15/low/111.png")
parser.add_argument('--result_path', type=str,  default="./result")
# model setting
parser.add_argument('--require_a',action='store_true')
parser.add_argument('--num_orders', type=int, default=6,help="orders of chebyfunc")
parser.add_argument('--num_fea', type=int, default=32)
parser.add_argument('--adb_type',type=str, default="triple")
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

assert opt != None, "opt is required !"
print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
# network
curvenet = mynet.ChebyAll3DAB(num_blocks=opt.num_orders, require_a=opt.require_a, num_feature=opt.num_fea, adb_type=opt.adb_type)
if cuda:
    curvenet = curvenet.cuda()
# summary(curvenet, torch.rand((1, 3, 256, 256)).cuda())
# transforms
trans_list = [
    transforms.ToTensor()
]
checkpoint = torch.load(opt.dir_pth)
curvenet.load_state_dict(checkpoint["net"])
os.makedirs(opt.result_path, exist_ok=True)
print("[*] RESULT will be saved in %s " % opt.result_path)
curvenet.eval()
for param in curvenet.parameters():
    param.requires_grad = False
print("[*] Model is READY.")
img_path_spilt = opt.img_path.split("/")
img_name = img_path_spilt[-1]
# data
img_PIL = Image.open(opt.img_path)
img_PIL = ((np.asarray(img_PIL) / 255.0) - 0.5) / 0.5
img_tensor = torch.from_numpy(img_PIL).float()
img_tensor = img_tensor.permute(2, 0, 1)
img_tensor = img_tensor.cuda().unsqueeze(0)
# test model
out = curvenet(img_tensor)
# save result
save_image(out, "%s/%s" % (opt.result_path, img_name))
torch.cuda.empty_cache()
print("[*] Test Finished ~ \n")
