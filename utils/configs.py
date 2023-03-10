# -*- coding: utf-8 -*-
import yaml
import os
import argparse

def get_configs_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--phase', type=str, default="train")
    parser.add_argument('--seed', type=int, default=903061)
    parser.add_argument('--num_orders', type=int, default=3)
    parser.add_argument('--config', help="configuration file",
                        type=str, default="configs/config")
    args = parser.parse_args()
    return args

def get_configs_from_yaml(file):
    file = file +".yaml"
    with open(file,"r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if "defaults" in cfg.keys():
        # You can define a defaults list in your primary Structured Config
        # just like you can in your primary config.yaml file.
        # please refer to https://hydra.cc/docs/1.0/tutorials/structured_config/defaults
        default_dit = cfg["defaults"]
        for key in default_dit:
            with open("configs/"+key+"/"+default_dit[key]+".yaml") as f:
                sub_cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg.update(sub_cfg)
    return cfg

def save_configs_as_yaml(opt=None,file=None):
    assert opt is not None,"[ * ] opt is a must !"
    if file is not None:
        file = file + ".yaml"
    else:
        file = os.path.join(opt.exp_path,opt.exp_name,"config.yaml")
    cfg = vars(opt)
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "w") as f:
        return yaml.dump(cfg,f)

def get_opt(opt):
    # opt = get_configs_from_args()
    cfg = get_configs_from_yaml(file=opt.config)
    opt = vars(opt)
    cfg.update(opt) 
    opt = argparse.Namespace(**cfg)
    return opt

def print_opt(opt,logger=None):
    '''
    :param opt: argparse object
    :return:  None
    '''
    opt_dict = vars(opt)
    if logger==None:
        print("*" *15+"config.yaml"+"*" *15)
        [print("%s: %s"%(k,opt_dict[k])) for k in opt_dict.keys()]
        print("*"*18)
    else:
        logger.info("*" *15+"config.yaml"+"*" *15)
        [logger.info("%s: %s" % (k, opt_dict[k])) for k in opt_dict.keys()]
        print("*" * 18)
basic_opt = get_configs_from_args()

