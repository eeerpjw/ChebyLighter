# -*- coding: utf-8 -*-
# files in the project
import networks.curve_models as mynet
# pytorch
import torch
import torch.optim

# torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchvision.utils import save_image

# utils packages
import os
import datetime
import time
from glob import glob
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from PIL import Image
from torchsummaryX import summary
# other files in the project
from networks.model_utils import weight_init_normal
import myloss
from metrics import MetricAll
from datapipeline.data_augmentation import DataAugmentation

class Model(object):
    def __init__(self, opt=None):
        super(Model, self).__init__()
        assert opt != None, "opt is required !"
        # print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))
        self.opt = opt
        self.cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        # network

        if opt.model=="ChebyAll3DAB":
            self.curvenet = mynet.ChebyAll3DAB(num_blocks=opt.num_orders, require_a=self.opt.require_a,num_feature=self.opt.num_fea, adb_type=self.opt.adb_type)

        if self.cuda:
            self.curvenet = self.curvenet.cuda()
        summary(self.curvenet, torch.rand((1, 3, 600, 400)).cuda())
        # data augmentation
        self.data_aug = DataAugmentation(self.opt.image_size)
        # creterion
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_l1 = torch.nn.L1Loss()
        self.criterion_ssim = myloss.SSIM()
        # self.criterion_tv = myloss.TVLoss(TVLoss_weight=2.0)
        # metrics
        self.metric_all = MetricAll()
        # tensorboard log
        tb_log_dir = os.path.join(opt.exp_path,opt.exp_name,"tb_runs")
        self.writer = SummaryWriter(log_dir=tb_log_dir, comment=opt.exp_name)

    def train(self, train_dataloader, eval_dataloader, logger=None):
        # model
        if self.opt.start_epoch != 0:
            checkpoint = self.load_checkpoint(ckp_name="cheby_%d" % self.opt.start_epoch)  # 不需要加.pth
            self.curvenet.load_state_dict(checkpoint["net"])
            optimizer = torch.optim.Adam(self.curvenet.parameters(), lr=self.opt.lr,
                                         betas=(self.opt.b1, self.opt.b2),
                                         weight_decay=self.opt.weight_decay)
            optimizer.load_state_dict(checkpoint["optimizer"])
            if self.opt.lr_scheduler=="exp":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.opt.lr_gamma)
            elif self.opt.lr_scheduler=="step":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                            step_size=self.opt.lr_step_size,
                                                            gamma=self.opt.lr_gamma)
            elif self.opt.lr_scheduler=="multistep":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                 milestones=[200, 300, 320, 340, 200],
                                                                 gamma=0.8)
            scheduler.load_state_dict(checkpoint["scheduler"])
            self.opt.start_epoch += 1 
        else:
            self.curvenet.apply(weight_init_normal)
            # optim
            optimizer = torch.optim.Adam(self.curvenet.parameters(), lr=self.opt.lr,
                                         betas=(self.opt.b1, self.opt.b2),
                                         weight_decay=self.opt.weight_decay)
            # optimizer
            if self.opt.lr_scheduler=="exp":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.opt.lr_gamma)
            elif self.opt.lr_scheduler=="step":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                            step_size=self.opt.lr_step_size,
                                                            gamma=self.opt.lr_gamma)
            elif self.opt.lr_scheduler=="multistep":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                 milestones=[200, 300, 320, 340, 200],
                                                                 gamma=0.8)
        vgg_loss = myloss.PerceptualLoss(self.opt)
        vgg_loss.cuda()
        vgg16 = myloss.load_vgg16('./pretrained_model')
        # vgg16.to(torch.device("cuda"))
        vgg16.eval()
        for param in vgg16.parameters():
            param.requires_grad = False

        prev_time = time.time()
        loss_dict_list = {"loss_ssim": [],
                          #"loss_tv": [],
                          "loss_vgg": [],
                          "loss_mse": [],
                          "loss_l1": [],
                          "loss_total": []
                          }
        metrics_dict_list = {"psnr": []}
        # start train
        best_psnr = 0.0
        for epoch in range(self.opt.start_epoch, self.opt.num_epoch):
            for i, sample in enumerate(train_dataloader):
                # data augmentation
                sample = self.data_aug(sample)
                input = Variable(sample["l"].type(self.Tensor))
                gt = Variable(sample["h"].type(self.Tensor))
                out = self.curvenet(input)

                loss_ssim = self.opt.w_ssim*(1 - self.criterion_ssim(out, gt))
                loss_vgg = self.opt.w_vgg * vgg_loss(vgg16, out, gt)
                loss_mse = self.opt.w_mse*self.criterion_mse(out,gt)
                loss_l1 = self.opt.w_l1*self.criterion_l1(out,gt)
                loss = loss_ssim + loss_mse + loss_vgg + loss_l1

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(net.parameters(), opt.grad_clip_norm)
                optimizer.step()

                # Determine approximate time left
                batches_done = epoch * len(train_dataloader) + i
                batches_left = self.opt.num_epoch * len(train_dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # tensorboard  log
                loss_dict = {"loss_ssim": loss_ssim.item(),
                             #"loss_tv": loss_tv.item(),
                             "loss_vgg": loss_vgg.item(),
                             "loss_mse": loss_mse.item(),
                             "loss_l1": loss_l1.item(),
                             "loss_total": loss.item()
                             }
                self.writer.add_scalar('train_loss/total_loss', loss.item(), epoch)
                self.writer.add_scalars('train_loss/every_single_loss', loss_dict, epoch)
                loss_dict["loss_total"] = loss.item()
                for key in loss_dict.keys():
                    loss_dict_list[key].append(loss_dict[key])

                logger.info(
                    "\r[Epoch %d/%d] [Batch %d/%d] [Lr %f] [loss: %f] [loss_ssim: %f] [loss_mse: %f] [loss_l1: %f] [loss_vgg: %f] ETA: %s"
                    % (epoch+1, self.opt.num_epoch, i+1, len(train_dataloader),
                       optimizer.state_dict()['param_groups'][0]['lr'],
                       loss.item(), loss_ssim.item(), loss_mse.item(), loss_l1.item(), loss_vgg.item(), time_left))
            psnr = self.evaluate(eval_dataloader, epoch,logger=logger)
            if psnr > best_psnr:
                checkpoint_ = {
                    "net": self.curvenet.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch
                }
                best_psnr = psnr
                self.save_checkpoint(checkpoint_, "cheby_best_%d_%0.3f"%(epoch,best_psnr))
            metrics_dict_list['psnr'].append(psnr)
            scheduler.step()

            if self.opt.ckp_freq != -1 and epoch % self.opt.ckp_freq == 0:
                # Save model checkpoints
                checkpoint_ = {
                    "net": self.curvenet.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch
                }
                self.save_checkpoint(checkpoint_, "cheby_%d" % epoch)
        self.writer.close()

        checkpoint_ = {
            "net": self.curvenet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch
        }
        self.save_checkpoint(checkpoint_, "cheby_latest")
        # Train finished
        self.plotloss(loss_dict_list)
        self.plotloss(metrics_dict_list,info='psnr')
        logger.info('\n[ * ]  Train finished ~ ')#, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        self.test(logger=logger)
        # self.testtime(logger=logger)
        self.test(logger=logger,best_psnr=best_psnr)

    def evaluate(self, eval_dataloader, epoch, logger=None):
        images_path = os.path.join(self.opt.exp_path, self.opt.exp_name,"result_eval")
        os.makedirs(images_path, exist_ok=True)
        logger.info('[ * ]  Evaluating %d images ' % (len(eval_dataloader) * self.opt.eval_bsz))#,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        # error_list = []
        self.curvenet.eval()
        time_consuming = 0
        for i, sample in enumerate(eval_dataloader):
            # to cuda tensor
            l_eval = Variable(sample["l"].type(self.Tensor), requires_grad=False)
            h_eval = Variable(sample["h"].type(self.Tensor), requires_grad=False)

            with torch.no_grad():
                ts = time.time()
                out = self.curvenet(l_eval)
                te = time.time()
                # error = self.criterion_ssim(out, h_eval)
                # error_list.append(error)

            img_sample = torch.cat((l_eval.data, out.data, h_eval.data), 0)
            save_image(img_sample, "%s/rst_%d_%d.png" % (images_path, epoch, i), nrow=self.opt.eval_bsz,
                       normalize=True)
            time_consuming += (te - ts)
            self.metric_all.compute_metrics(h_eval, out)
            torch.cuda.empty_cache()
        self.curvenet.train()
        time_consuming = time_consuming / len(eval_dataloader) / self.opt.eval_bsz
        # tensorboard  log
        # self.writer.add_scalar('eval/ssim', error_list, epoch)
        self.metric_all.compute_statics()
        # logger.info("\r psnr for each test image:",self.metric_all.psnr_list)
        self.writer.add_scalars('eval/psnr', self.metric_all.psnr_statics, epoch)
        self.writer.add_scalar('eval/time comsuming', time_consuming, epoch)
        logger.info("\r evaluate finished [PSNR %f] [Average time per image: %fs]"
              % (self.metric_all.psnr_statics["mean"], time_consuming))
        psnr = self.metric_all.psnr_statics["mean"]
        self.metric_all.clear_all()

        # return np.mean(error_list)
        return psnr

    def test(self, logger=None, best_psnr=None):

        image_list = []
        for ext in ('.bmp', '.png', '.jpg', '.JPG', '.jpeg'):
            image_list.extend(glob(self.opt.test_path + "/*/*" + ext))
            # image_list.extend(glob(test_path + "/*" + ext))
        logger.info("[*] Test data from %s " % (self.opt.test_path + "/*/*"))
        imgs_num = len(image_list)
        logger.info("[*] Got %d images for test" % imgs_num)
        # transforms
        trans_list = [
            transforms.ToTensor()
        ]
        if self.opt.test_epoch==-1 and best_psnr is None:
            checkpoint = self.load_checkpoint(ckp_name="cheby_latest")
            result_path = os.path.join(self.opt.exp_path, self.opt.exp_name,"result_test_latest")
        elif best_psnr is None:
            checkpoint = self.load_checkpoint(ckp_name="cheby_%d" % self.opt.test_epoch)
            result_path =  os.path.join(self.opt.exp_path, self.opt.exp_name,"result_test_%d"%self.opt.test_epoch)
        elif best_psnr is not None:
            checkpoint = self.load_checkpoint(ckp_name="cheby_best_%0.3f"%best_psnr)
            result_path = os.path.join(self.opt.exp_path, self.opt.exp_name, "result_test_best_%0.3f"%best_psnr)
        self.curvenet.load_state_dict(checkpoint["net"])
        for d in glob(os.path.join(self.opt.test_path,"*")):
            savepath = os.path.join(result_path, d.split('/')[-1])
            os.makedirs(savepath, exist_ok=True)
        logger.info("[*] RESULT will be saved in %s " % result_path)
        # curve_net.eval()
        for param in self.curvenet.parameters():
            param.requires_grad = False
        logger.info("[*] Model is READY.")
        self.curvenet.eval()
        prev_time = time.time()
        for i, img_path in enumerate(image_list):
            img_path_spilt = img_path.split("/")
            dataset_name, img_name = img_path_spilt[-2], img_path_spilt[-1]
            img_PIL = Image.open(img_path)
            img_PIL = ((np.asarray(img_PIL) / 255.0) - 0.5) / 0.5
            img_tensor = torch.from_numpy(img_PIL).float()
            img_tensor = img_tensor.permute(2, 0, 1)
            img_tensor = img_tensor.cuda().unsqueeze(0)

            if self.opt.require_a:
                Y, A, I = self.curvenet(img_tensor)
                out = Y[-1]
                A = self.normalizeto01(A)
                save_image(out, "%s/%s/%s" % (result_path, dataset_name, img_name))
                x = torch.cat(Y, 0)
                a = torch.cat(A, 0)
                ai = torch.cat(I, 0)
                save_image(x, "%s/%s/Y_%s" % (result_path, dataset_name, img_name))
                save_image(a, "%s/%s/A_%s" % (result_path, dataset_name, img_name))
                save_image(ai, "%s/%s/I_%s" % (result_path, dataset_name, img_name))
                del Y, A, I
            else:
                out = self.curvenet(img_tensor)
                save_image(out, "%s/%s/%s" % (result_path, dataset_name, img_name))
            torch.cuda.empty_cache()
            # Determine approximate time left
            num_left = imgs_num - i
            time_left = datetime.timedelta(seconds=num_left * (time.time() - prev_time))
            prev_time = time.time()
            logger.info(
                "\r[Testing %d/%d]  %s %s ETA: %s"% (i, imgs_num, dataset_name, img_name, time_left)
            )

        logger.info("[*] Test Finished ~ \n")

    def save_checkpoint(self, checkpoint, ckp_name):
        """
        :param checkpoint: dict include net params, optimizer, epoch
        :return: None
        """
        print("[ * ] Svaing checkpoint : %s"%ckp_name)
        save_path = os.path.join(self.opt.exp_path, self.opt.exp_name, "checkpoint/%s.pth" % ckp_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, ckp_name):
        """
        :param ckp_name:
        :param epoch:
        :return:
        """
        load_name = os.path.join(self.opt.exp_path, self.opt.exp_name, "checkpoint/%s.pth" % ckp_name)
        return torch.load(load_name)

    def plotloss(self, loss_dict_list, info=None):
        for key in loss_dict_list.keys():
            plt.plot(range(len(loss_dict_list[key])), loss_dict_list[key], linewidth=1, label=key)
        plt.legend()
        plt.xlabel("iterations")

        save_path = os.path.join(self.opt.exp_path, self.opt.exp_name)
        os.makedirs(save_path, exist_ok=True)
        if info is not None:
            savename = '%s/loss_%s_%s.svg' % (save_path, self.opt.exp_name,info)
        else:
            savename = '%s/loss_%s.svg' % (save_path, self.opt.exp_name)
        plt.savefig(savename, dpi=1200, format='svg')
        plt.close()

    def normalizeto01(self, tesnsor_lst):
        out_lst = []
        for i, x in enumerate(tesnsor_lst):
            out_lst.append(x * 0.5 + 0.5)
        return out_lst

    def testtime(self, logger=None):

        testdata_path = '../LOLRealow100/*'
        logger.info("[*] Test data from %s " % testdata_path)
        image_list = glob(testdata_path)
        imgs_num = len(image_list)
        logger.info("[*] Got %d images for test" % imgs_num)
        # 定义transforms
        if self.opt.test_epoch==-1:
            checkpoint = self.load_checkpoint(ckp_name="cheby_latest")
            result_path = os.path.join(self.opt.exp_path, self.opt.exp_name,"result_test_latest")
        else:
            checkpoint = self.load_checkpoint(ckp_name="cheby_%d" % self.opt.test_epoch)
            result_path =  os.path.join(self.opt.exp_path, self.opt.exp_name,"result_test_%d"%self.opt.test_epoch)
        self.curvenet.load_state_dict(checkpoint["net"])

        savepath = os.path.join(result_path, 'fivek_test')
        os.makedirs(savepath, exist_ok=True)
        logger.info("[*] RESULT will be saved in %s " % savepath)
        # curve_net.eval()
        for param in self.curvenet.parameters():
            param.requires_grad = False

        logger.info("[*] Model is READY.")

        total_time = 0.0
        for i, img_path in enumerate(image_list):
            img_path_spilt = img_path.split("/")
            img_name = img_path_spilt[-1]
            img_PIL = Image.open(img_path)
            img_PIL = ((np.asarray(img_PIL) / 255.0) - 0.5) / 0.5
            img_tensor = torch.from_numpy(img_PIL).float()
            img_tensor = img_tensor.permute(2, 0, 1)
            img_tensor = img_tensor.cuda().unsqueeze(0)
            start_time = time.time()
            out = self.curvenet(img_tensor)
            end_time = time.time()
            logger.info("{} {} s".format(img_name, end_time - start_time))
            total_time += end_time - start_time
            save_image(out, "%s/%s" % (result_path, img_name))
            torch.cuda.empty_cache()

        average_time = total_time / 100.0
        logger.info("total: {} s, average: {} s".format(total_time, average_time))