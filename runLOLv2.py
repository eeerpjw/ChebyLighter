# -*- coding: utf-8 -*-

import os
import torch
# utils packages
from utils.configs import get_opt,basic_opt,save_configs_as_yaml
os.environ["CUDA_VISIBLE_DEVICES"] = basic_opt.gpu_id
from utils.logger import get_logger
from datapipeline.LOLv2 import LOLv2Dataset
from model.modelLOL import Model
from datetime import datetime
# set random seed
import random
import numpy as np
random.seed(basic_opt.seed)
np.random.seed(basic_opt.seed)
torch.manual_seed(basic_opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(basic_opt.seed)

def main(basic_opt):
    opt = get_opt(basic_opt)
    if opt.phase=="train":
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        if opt.comment != "None":
            opt.exp_name = opt.exp_name+"_"+opt.model+"_%d"%opt.num_orders+"_"+opt.machine_id+"_"+opt.comment+"_"+current_time
        else:
            opt.exp_name = opt.exp_name+"_"+opt.model+"_%d"%opt.num_orders+"_"+opt.machine_id+"_"+ current_time
        save_configs_as_yaml(opt=opt,
                                 file=os.path.join(opt.exp_path,opt.exp_name,"config"))
    logger = get_logger(level=opt.logger_level,
                        log_file=os.path.join(opt.exp_path,opt.exp_name,"log_%s.log"%opt.phase))
    logger.info("ExpName: %s"%opt.exp_name)
    opt.data_path = os.path.join(opt.data_path, opt.dataset)
    # opt.test_path = os.path.join(opt.data_path,opt.dataset, opt.testset)
    model = Model(opt)
    if opt.phase=="train":
        print("train")

        train_data = LOLv2Dataset(data_dir=opt.data_path, mode='train')
        train_dataloader = torch.utils.data.DataLoader(train_data,
                                                       batch_size=opt.train_bsz,
                                                       shuffle=True,
                                                       num_workers=opt.num_workers,
                                                       pin_memory=True)
        eval_data = LOLv2Dataset(data_dir=opt.data_path, mode='eval')
        eval_dataloader = torch.utils.data.DataLoader(eval_data,
                                                  batch_size=opt.eval_bsz,
                                                  shuffle=False,
                                                  num_workers=opt.num_workers,
                                                  pin_memory=True)
        model.train(train_dataloader, eval_dataloader,logger=logger)
    if opt.phase=="test":
        model.test(logger=logger)

if __name__ == "__main__":
    main(basic_opt)