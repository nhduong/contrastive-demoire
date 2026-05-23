'''
    This file contains all the functions that are used in the main file.
'''

import hydra
from hydra.utils import instantiate, get_original_cwd, to_absolute_path
from omegaconf import DictConfig

from termcolor import colored

import os
import math
from enum import Enum
import random
import platform

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from natsort import natsorted
from glob import glob
import torchvision.models as models
import torch.nn.functional as F

import numpy as np
from PIL import Image
from PIL import ImageFile

from tqdm import tqdm
import datetime
import sys

from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim
from math import log10
from math import exp
import cv2

import subprocess
import logging
import traceback
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")
warnings.filterwarnings("ignore", message="Warning: find_unused_parameters=True")
warnings.filterwarnings("ignore", message="[matplotlib.image][WARNING] - Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).")
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore")

from torchvision.utils import save_image

import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from fvcore.nn import FlopCountAnalysis

import time
import subprocess

import pkg_resources
from torch.utils.collect_env import get_pretty_env_info
import psutil

from safetensors.torch import load_file


def train(data_loader, demo_net, accelerator, device, criterion_1, criterion_2, optimizer, epoch, args, t0, lr, compute_metrics):
    # AverageMeter is used to track performance
    total_loss_meter = AverageMeter()
    l1_loss_meter = AverageMeter()
    per_loss_meter = AverageMeter()
    cont_loss_meter = AverageMeter()
    
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
        
    # init metrics
    l1_loss_1_tensor = torch.tensor(0.0).to(device)
    l1_loss_2_tensor = torch.tensor(0.0).to(device)
    l1_loss_3_tensor = torch.tensor(0.0).to(device)
    l1_loss_total_tensor = torch.tensor(0.0).to(device)
    per_loss_tensor = torch.tensor(0.0).to(device)
    cont_loss_tensor = torch.tensor(0.0).to(device)

    psnr_tensor = torch.tensor(0.0).to(device)
    ssim_tensor = torch.tensor(0.0).to(device)

    # switch to train mode
    demo_net.train()

    # double-check the number of samples in the dataloader
    data_loader_len = (len(data_loader) * args.data.train.batch_size) * accelerator.num_processes
    proc_items = 0

    if accelerator.is_main_process:
        print(">>> number of train samples (pairs): ", data_loader_len)
        
    # progress bar
    if accelerator.is_main_process:
        train_time = get_time_diff(t0, datetime.datetime.now())
        pbar = tqdm(
            total=data_loader_len,
            dynamic_ncols=True,
            bar_format= "t:{percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}] {desc}",
            desc="E:%03d | P:%s | S:%s | lr:%.3e | L:%.2f | L1:%s | Lp:%s | Lc:%s (%s)" % \
                (
                    epoch,
                    colored(psnr_meter.avg, "green"),
                    colored("{:.4f}".format(ssim_meter.avg), "green"),
                    optimizer.param_groups[0]["lr"],
                    total_loss_meter.avg,
                    colored("{:.2f}".format(l1_loss_meter.avg), "red"),
                    colored("{:.2f}".format(per_loss_meter.avg), "red"),
                    colored("{:.4f}".format(cont_loss_meter.avg), "red"),
                    train_time,
                ), \
            ascii="-░▒█",
        )

    # ---------------------------------------------------
    # training loop
    # ---------------------------------------------------
    for i_batch, data in enumerate(data_loader):
        # ---------------------------------------------------
        # get the data
        # ---------------------------------------------------
        I_1 = data["moire"]
        P_1 = data["clean"]
        
        # multi-scale ground truths
        P_2 = F.interpolate(P_1, scale_factor=0.5, mode='bilinear', align_corners=False)
        P_3 = F.interpolate(P_1, scale_factor=0.25, mode='bilinear', align_corners=False)

        # multi-scale inputs
        P_2 = F.interpolate(I_1, scale_factor=0.5, mode='bilinear', align_corners=False)
        P_3 = F.interpolate(I_1, scale_factor=0.25, mode='bilinear', align_corners=False)
        
        # ---------------------------------------------------
        # compute the outputs
        # ---------------------------------------------------
        C_1, C_2, C_3, M_1, M_2, M_3 = demo_net(I_1)
                
        # ---------------------------------------------------
        # contrastive loss with better lower bounds
        # ---------------------------------------------------
        if args.losses.contrastive.use:
            if args.losses.contrastive.base == "l1":
                cont_loss_tensor = tcsvt_contrastive_loss(criterion_1, C_1, C_2, C_3, P_1, P_2, P_3, M_1, args)
            elif args.losses.contrastive.base == "perceptual":
                cont_loss_tensor = tcsvt_contrastive_loss(criterion_2, C_1, C_2, C_3, P_1, P_2, P_3, M_1, args)
                                            
        # ---------------------------------------------------
        # l1 norm
        # ---------------------------------------------------
        if args.losses.l1.use:
            # l-1 norm only
            l1_loss_1_tensor = criterion_1(C_1, P_1)
            l1_loss_2_tensor = criterion_1(C_2, P_2)
            l1_loss_3_tensor = criterion_1(C_3, P_3)

            # <!> LOG
            if args.losses.consistency.log:
                if args.losses.contrastive.use and args.losses.consistency.l1_weight > 0:
                    # consistency loss
                    l1_loss_1_tensor += args.losses.consistency.l1_weight * criterion_1(C_1 + M_1, I_1) * math.log(1.0 * lr / args.opt.eta_min, args.opt.lr/args.opt.eta_min)
                    l1_loss_2_tensor += args.losses.consistency.l1_weight * criterion_1(C_2 + M_2, P_2) * math.log(1.0 * lr / args.opt.eta_min, args.opt.lr/args.opt.eta_min)
                    l1_loss_3_tensor += args.losses.consistency.l1_weight * criterion_1(C_3 + M_3, P_3) * math.log(1.0 * lr / args.opt.eta_min, args.opt.lr/args.opt.eta_min)
            else:
                if args.losses.contrastive.use and args.losses.consistency.l1_weight > 0:
                    # consistency loss
                    l1_loss_1_tensor += args.losses.consistency.l1_weight * criterion_1(C_1 + M_1, I_1)
                    l1_loss_2_tensor += args.losses.consistency.l1_weight * criterion_1(C_2 + M_2, P_2)
                    l1_loss_3_tensor += args.losses.consistency.l1_weight * criterion_1(C_3 + M_3, P_3)

        l1_loss_total_tensor = l1_loss_1_tensor + l1_loss_2_tensor + l1_loss_3_tensor
        l1_loss_total_tensor = args.losses.l1.weight * l1_loss_total_tensor

        # ---------------------------------------------------
        # perceptual loss
        # ---------------------------------------------------
        if args.losses.perceptual.use:
            per_loss_tensor = criterion_2(C_1, P_1, feature_layers=[2]) + \
                criterion_2(C_2, P_2, feature_layers=[2]) + \
                    criterion_2(C_3, P_3, feature_layers=[2])
                
            if args.losses.contrastive.use and args.losses.consistency.perceptual_weight > 0:
                # consistency loss
                per_loss_tensor += args.losses.consistency.perceptual_weight * (
                    criterion_2(C_1 + M_1, I_1, feature_layers=[2]) + \
                        criterion_2(C_2 + M_2, P_2, feature_layers=[2]) + \
                            criterion_2(C_3 + M_3, P_3, feature_layers=[2])
                )
                
            per_loss_tensor = args.losses.perceptual.weight * per_loss_tensor
                        
        # ---------------------------------------------------
        # total loss
        # ---------------------------------------------------
        
        # adaptive loss balancing based on gradients of the last kayers
        params = list(demo_net.recon_0_clean.parameters()) + list(demo_net.recon_1_clean.parameters()) + list(demo_net.recon_2_clean.parameters())

        # ==== g1 ====
        grads1 = torch.autograd.grad(
            l1_loss_total_tensor,
            params,
            retain_graph=True,
            allow_unused=True
        )

        vec1 = get_grad_vector(grads1)
        g1 = vec1.norm()

        # ==== g2 ====
        grads2 = torch.autograd.grad(
            per_loss_tensor,
            params,
            retain_graph=True,
            allow_unused=True
        )

        vec2 = get_grad_vector(grads2)
        g2 = vec2.norm()

        # ==== g3 ====
        grads3 = torch.autograd.grad(
            cont_loss_tensor,
            params,
            retain_graph=True,
            allow_unused=True
        )

        vec3 = get_grad_vector(grads3)
        g3 = vec3.norm()
                
        cos12 = F.cosine_similarity(vec1, vec2, dim=0)
        cos13 = F.cosine_similarity(vec1, vec3, dim=0)
        # cos23 = F.cosine_similarity(vec2, vec3, dim=0)
        
        lambda2 = (g1 / (g2 + 1e-8)).detach()
        lambda3 = (g1 / (g3 + 1e-8)).detach()

        lambda2 = lambda2 * torch.clamp(cos12, min=0.0)
        lambda3 = lambda3 * torch.clamp(cos13, min=0.0)
        
        # losses with weighted balancing
        per_loss_tensor = lambda2 * per_loss_tensor
        cont_loss_tensor = lambda3 * cont_loss_tensor
        
        # <!> LOG
        if args.losses.contrastive.log:
            cont_loss_tensor = cont_loss_tensor * math.log(1.0 * lr / args.opt.eta_min, args.opt.lr/args.opt.eta_min)

        loss = l1_loss_total_tensor + per_loss_tensor + cont_loss_tensor
        
        # ---------------------------------------------------
        # optimization step
        # ---------------------------------------------------
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # backward pass
        if args.utils.hf_accelerate:
            # accelerator.backward(loss, retain_graph=True)
            accelerator.backward(loss)
        else:
            loss.backward()

        # gradient clipping
        if args.opt.grad_clip > 0:
            accelerator.clip_grad_norm_(
                list(demo_net.parameters()),
                args.opt.grad_clip
            )
            
        # update weights
        optimizer.step()

        # ---------------------------------------------------
        # training logs
        # ---------------------------------------------------
        with torch.no_grad():
            # measure PSNR and SSIM
            psnr_tensor = ssim_tensor = torch.tensor(0.0).to(device)
            if not args.exp.dont_calc_mets_at_all and args.exp.calc_train_mets:
                if (epoch + 1) % round(args.opt.T_0 / 2) == 0 or epoch % args.opt.T_0 == 0 or args.exp.evaluate:
                    try:
                        psnr_tensor, ssim_tensor = compute_metrics.compute(C_1, P_1, eval_name=None)
                    except Exception as e:
                        if accelerator.is_main_process:
                            print(">> error in metrics calculation")
                            print(e)
                        pass
            
            # gather metrics
            loss = torch.mean(accelerator.gather_for_metrics(loss))
            
            l1_loss_total_tensor = torch.mean(accelerator.gather_for_metrics(l1_loss_total_tensor))
            per_loss_tensor = torch.mean(accelerator.gather_for_metrics(per_loss_tensor))
            cont_loss_tensor = torch.mean(accelerator.gather_for_metrics(cont_loss_tensor))
            
            psnr_tensor = torch.mean(accelerator.gather_for_metrics(psnr_tensor))
            ssim_tensor = torch.mean(accelerator.gather_for_metrics(ssim_tensor))
                        
            # update meters
            num_samples = 1 # mean already taken
            total_loss_meter.update(loss.item(), num_samples)
            l1_loss_meter.update(l1_loss_total_tensor.item(), num_samples)
            cont_loss_meter.update(cont_loss_tensor.item(), num_samples)
            per_loss_meter.update(per_loss_tensor.item(), num_samples)
            
            try:
                psnr_meter.update(psnr_tensor.item(), num_samples)
                ssim_meter.update(ssim_tensor.item(), num_samples)
            except:
                psnr_meter.update(psnr_tensor, num_samples)
                ssim_meter.update(ssim_tensor, num_samples)
            
            # update progress bar
            proc_items += I_1.size(0) * accelerator.num_processes
            if data_loader_len > 0:
                if (i_batch == data_loader_len - 1) or (i_batch % max(5, round(data_loader_len / args.utils.print_freq)) == 0):
                    if accelerator.is_main_process:
                        train_time = get_time_diff(t0, datetime.datetime.now())
                        pbar.set_description_str(
                            desc = "E:%03d | P:%s | S:%s | lr:%.3e | L:%.2f | L1:%s | Lp:%s | Lc:%s (%s)" % \
                                (
                                    epoch,
                                    colored("{:.4f}".format(psnr_meter.avg), "green"),
                                    colored("{:.4f}".format(ssim_meter.avg), "green"),
                                    optimizer.param_groups[0]["lr"],
                                    total_loss_meter.avg,
                                    colored("{:.2f}".format(l1_loss_meter.avg), "red"),
                                    colored("{:.2f}".format(per_loss_meter.avg), "red"),
                                    colored("{:.4f}".format(cont_loss_meter.avg), "red"),
                                    train_time
                                ),
                            refresh=False,
                        )
                        pbar.update(proc_items)

                    proc_items = 0

            # skip for debug
            if args.utils.try_it:
                break

    if accelerator.is_main_process:
        pbar.update(pbar.total - pbar.n)
        pbar.close()

    return total_loss_meter.avg, l1_loss_meter.avg, per_loss_meter.avg, cont_loss_meter.avg, psnr_meter.avg, ssim_meter.avg


def validate(data_loader, demo_net, device, accelerator, epoch, args, compute_metrics, t0):
    if accelerator.is_main_process:
        if args.data.val.patch_size:
            print("\n=====================\n>>> WARNING: using VAL_PATCH_SIZE for fast validation\n\tfor ACCURATE results, use full images\n=====================\n")
        if args.exp.fast_psnr and args.utils.eval.name == None:
            print("\n=====================\n>>> WARNING: using FAST_PSNR for fast validation\n\tfor ACCURATE results, modify fast_psnr in the config file\n=====================\n")
        else:
            print('\n\n==================\nargs.utils.eval.name: ', args.utils.eval.name, '\n==================\n\n')

    # save images for qualitative evaluation
    if accelerator.is_main_process:
        if args.exp.save_tests != "":
            os.makedirs(args.exp.save_tests, exist_ok=True)
            os.makedirs(os.path.join(args.exp.save_tests, str(epoch)), exist_ok=True)
            print(">>> save tests to: ", args.exp.save_tests)
    
    # double-check the number of images
    data_loader_len = len(data_loader) * args.data.val.batch_size * accelerator.num_processes
    proc_items = 0

    # evaluation meters
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    # init metrics
    psnr_tensor = torch.tensor(0.0).to(device)
    ssim_tensor = torch.tensor(0.0).to(device)
    
    # switch to evaluate mode
    demo_net.eval()

    # validation loop
    with torch.no_grad():
        if accelerator.is_main_process:
            train_time = get_time_diff(t0, datetime.datetime.now())
            
            pbar = tqdm(
                total=data_loader_len,
                dynamic_ncols=True,
                bar_format="v:{percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}] {desc}",
                desc="E:%03d | P:%.4f | S:%.4f (%s)" % 
                    (
                        epoch,
                        psnr_meter.avg,
                        ssim_meter.avg,
                        train_time,
                    ),
                ascii="-░▒█",
            )

        for i_batch, data in enumerate(data_loader):
            # update progress bar
            proc_items += args.data.val.batch_size * accelerator.num_processes

            # get the inputs
            I_1 = data["moire"]
            P_1 = data["clean"]
            
            # get image size
            _, _, h_1, w_1 = I_1.size()

            # pad image such that the resolution is a multiple of 32 for multi-scale processing
            w_pad_1 = (math.ceil(w_1/32)*32 - w_1) // 2
            h_pad_1 = (math.ceil(h_1/32)*32 - h_1) // 2
            I_1 = img_pad(I_1, w_r=w_pad_1, h_r=h_pad_1)

            # ==============================
            # compute outputs
            # ==============================
            C_1, _, _, M_1, _, _ = demo_net(I_1)
            
            # padding removal
            if h_pad_1 != 0:
                C_1 = C_1[:, :, h_pad_1:-h_pad_1, :]
                M_1 = M_1[:, :, h_pad_1:-h_pad_1, :]
            if w_pad_1 != 0:
                C_1 = C_1[:, :, :, w_pad_1:-w_pad_1]
                M_1 = M_1[:, :, :, w_pad_1:-w_pad_1]
            
            # measure PSNR and SSIM
            if not args.exp.dont_calc_mets_at_all:
                if (epoch + 1) % round(args.opt.T_0 / 2) == 0 or epoch % args.opt.T_0 == 0 or args.exp.evaluate:
                    try:
                        psnr_tensor, ssim_tensor = compute_metrics.compute(C_1, P_1, eval_name=args.utils.eval.name)
                    except Exception as e:
                        if accelerator.is_main_process:
                            print(">> error in metrics calculation")
                            print(e)
                        pass

            # gather metrics
            psnr_tensor = torch.mean(accelerator.gather_for_metrics(psnr_tensor))
            ssim_tensor = torch.mean(accelerator.gather_for_metrics(ssim_tensor))

            # update meters
            num_samples = 1 # mean already taken
            psnr_meter.update(psnr_tensor, num_samples)
            ssim_meter.update(ssim_tensor, num_samples)

            # save test images
            if args.exp.save_tests != "":
                for img_id in range(I_1.size(0)):
                    # extract file title
                    fn_only = os.path.splitext(os.path.basename(data["moire_fn"][img_id]))[0]
                    
                    # demoired images
                    save_image(torch.clamp(C_1[img_id], 0, 1), os.path.join(args.exp.save_tests, str(epoch), fn_only + "_demoired.jpg"))
            
            # update progress bar
            if data_loader_len > 0:
                if (i_batch == data_loader_len - 1) or (i_batch % max(5, round(data_loader_len / args.utils.print_freq)) == 0):
                    if accelerator.is_main_process:
                        train_time = get_time_diff(t0, datetime.datetime.now())
                        
                        pbar.set_description_str(
                            "E:%03d | P:%.4f | S:%.4f (%s)" % 
                                (
                                    epoch,
                                    psnr_meter.avg,
                                    ssim_meter.avg,
                                    train_time
                                ), 
                            refresh=False,
                        )
                        pbar.update(proc_items)

                    proc_items = 0

            # early stopping
            if args.utils.try_it:
                break
        
        if accelerator.is_main_process:
            pbar.update(pbar.total - pbar.n)
            pbar.close()
            
            print('\n## val PSNR {psnr.avg:.5f} | val SSIM {ssim.avg:.5f} @ epoch {epoch}\n'.format(psnr=psnr_meter, ssim=ssim_meter, epoch=epoch))

    return psnr_meter.avg, ssim_meter.avg


# ---------------------------------------------
# metrics
# ---------------------------------------------
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    """
    Fast pytorch implementation for SSIM, referred from
    "https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py"
    """
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
        
        
class PSNR(torch.nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, img1, img2):
        psnr = -10*torch.log10(torch.mean((img1-img2)**2))
        
        return psnr
    
    
"""
A pytorch implementation for reproducing results in MATLAB, slightly modified from
https://github.com/mayorx/matlab_ssim_pytorch_implementation.
"""
def generate_1d_gaussian_kernel():
    return cv2.getGaussianKernel(11, 1.5)

def generate_2d_gaussian_kernel():
    kernel = generate_1d_gaussian_kernel()
    return np.outer(kernel, kernel.transpose())

def generate_3d_gaussian_kernel():
    kernel = generate_1d_gaussian_kernel()
    window = generate_2d_gaussian_kernel()
    return np.stack([window * k for k in kernel], axis=0)

class MATLAB_SSIM(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(MATLAB_SSIM, self).__init__()
        self.device = device
        conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
        conv3d.weight.requires_grad = False
        conv3d.weight[0, 0, :, :, :] = torch.tensor(generate_3d_gaussian_kernel())
        self.conv3d = conv3d.to(device)
        # self.conv3d = conv3d

        conv2d = torch.nn.Conv2d(1, 1, (11, 11), stride=1, padding=(5, 5), bias=False, padding_mode='replicate')
        conv2d.weight.requires_grad = False
        conv2d.weight[0, 0, :, :] = torch.tensor(generate_2d_gaussian_kernel())
        self.conv2d = conv2d.to(device)
        # self.conv2d = conv2d

    def forward(self, img1, img2):
        assert len(img1.shape) == len(img2.shape)
        with torch.no_grad():
            img1 = torch.tensor(img1).to(self.device).float()
            img2 = torch.tensor(img2).to(self.device).float()
            # img1 = torch.tensor(img1).float()
            # img2 = torch.tensor(img2).float()

            if len(img1.shape) == 2:
                conv = self.conv2d
            elif len(img1.shape) == 3:
                conv = self.conv3d
            else:
                raise not NotImplementedError('only support 2d / 3d images.')
            return self._ssim(img1, img2, conv)

    def _ssim(self, img1, img2, conv):
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu1 = conv(img1)
        mu2 = conv(img2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = conv(img1 ** 2) - mu1_sq
        sigma2_sq = conv(img2 ** 2) - mu2_sq
        sigma12 = conv(img1 * img2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) *
                    (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                           (sigma1_sq + sigma2_sq + C2))

        return float(ssim_map.mean())


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])

    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = torchvision.utils.make_grid(tensor, nrow=int(math.sqrt(n_img)), padding=0, normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    
    
    return img_np.astype(out_type)


class create_metrics():
    """
       We note that for different benchmarks, previous works calculate metrics in different ways, which might
       lead to inconsistent SSIM results (and slightly different PSNR), and thus we follow their individual
       ways to compute metrics on each individual dataset for fair comparisons.
       For our 4K dataset, calculating metrics for 4k image is much time-consuming,
       thus we benchmark evaluations for all methods with a fast pytorch SSIM implementation referred from
       "https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py".
    """
    def __init__(self, args, device, accelerator=None):
        self.data_type = args.data.name
        self.fast_ssim = SSIM()
        self.fast_psnr = PSNR()
        self.matlab_ssim = MATLAB_SSIM(device=device)
        self.device = device
        self.args = args
        
        if args.exp.fast_psnr:
            print('\n\n==================\nevaluation metrics: FAST\n==================\n\n')
        else:
            if self.data_type == 'uhdm':
                print('\n\n==================\nevaluation metrics: FAST\n==================\n\n')
            elif self.data_type == 'fhdmi':
                print('\n\n==================\nevaluation metrics: fhdmi\n==================\n\n')
            elif self.data_type == 'tip18':
                print('\n\n==================\nevaluation metrics: tip18\n==================\n\n')
            elif self.data_type == 'aim':
                print('\n\n==================\nevaluation metrics: aim\n==================\n\n')
            else:
                print('\n\n==================\nevaluation metrics: Unrecognized data_type for evaluation!\n==================\n\n')

    def compute(self, out_img, gt, eval_name=None):
        if eval_name:
            if eval_name == 'uhdm':
                res_psnr, res_ssim = self.fast_psnr_ssim(out_img, gt)
            elif eval_name == 'fhdmi':
                res_psnr, res_ssim = self.skimage_psnr_ssim(out_img, gt)
            elif eval_name == 'tip18':
                res_psnr, res_ssim = self.matlab_psnr_ssim(out_img, gt)
            elif eval_name == 'aim':
                res_psnr, res_ssim = self.aim_psnr_ssim(out_img, gt)
            else:
                print('Unrecognized data_type for evaluation!')
                raise NotImplementedError
        else:
            if self.args.exp.fast_psnr:
                res_psnr, res_ssim = self.fast_psnr_ssim(out_img, gt)
            else:
                if self.data_type == 'uhdm':
                    res_psnr, res_ssim = self.fast_psnr_ssim(out_img, gt)
                elif self.data_type == 'fhdmi':
                    res_psnr, res_ssim = self.skimage_psnr_ssim(out_img, gt)
                elif self.data_type == 'tip18':
                    res_psnr, res_ssim = self.matlab_psnr_ssim(out_img, gt)
                elif self.data_type == 'aim':
                    res_psnr, res_ssim = self.aim_psnr_ssim(out_img, gt)
                else:
                    print('Unrecognized data_type for evaluation!')
                    raise NotImplementedError
        
        # check if res_psnr/res_ssim is a tensor
        if not torch.is_tensor(res_psnr):
            res_psnr = torch.tensor(res_psnr, device=self.device)
        if not torch.is_tensor(res_ssim):
            res_ssim = torch.tensor(res_ssim, device=self.device)
            
        return res_psnr, res_ssim


    def fast_psnr_ssim(self, out_img, gt):
        pre = torch.clamp(out_img, min=0, max=1)
        tar = torch.clamp(gt, min=0, max=1)
        psnr = self.fast_psnr(pre, tar)
        ssim = self.fast_ssim(pre, tar)
        return psnr, ssim

    def skimage_psnr_ssim(self, out_img, gt):
        """
        Same with the previous SOTA FHDe2Net: https://github.com/PKU-IMRE/FHDe2Net/blob/main/test.py
        """
        mi1 = tensor2img(out_img)
        mt1 = tensor2img(gt)
        psnr = ski_psnr(mt1, mi1)
        ssim = ski_ssim(mt1, mi1, multichannel=True, channel_axis=2, data_range=255)
        return psnr, ssim

    def matlab_psnr_ssim(self, out_img, gt):
        """
        A pytorch implementation for reproducing SSIM results when using MATLAB
        same with the previous SOTA MopNet: https://github.com/PKU-IMRE/MopNet/blob/master/test_with_matlabcode.m
        """
        mi1 = tensor2img(out_img)
        mt1 = tensor2img(gt)
        psnr = ski_psnr(mt1, mi1)
        ssim = self.matlab_ssim(mt1, mi1)
        return psnr, ssim

    def aim_psnr_ssim(self, out_img, gt):
        """
        Same with the previous SOTA MBCNN: https://github.com/zhenngbolun/Learnbale_Bandpass_Filter/blob/master/main_multiscale.py
        """
        mi1 = tensor2img(out_img)
        mt1 = tensor2img(gt)
        mi1 = mi1.astype(np.float32) / 255.0
        mt1 = mt1.astype(np.float32) / 255.0
        psnr = 10 * log10(1 / np.mean((mt1 - mi1) ** 2))
        ssim = ski_ssim(mt1, mi1, multichannel=True, channel_axis=2, data_range=1)
        return psnr, ssim


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, fmt=':f', summary_type=Summary.AVERAGE):
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        # self.sum += val * n
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, l1_loss=None, resize=True, blocks=[]):
        super(VGGPerceptualLoss, self).__init__()
        self.blocks = blocks
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize
        self.l1_loss = l1_loss

    # def forward(self, moire_img, clean_img, feature_layers=[0, 1, 2, 3], style_layers=[], use_fft=False):
    def forward(self, moire_img, clean_img, feature_layers=[2], style_layers=[], use_fft=False):
        if moire_img.shape[1] != 3:
            moire_img = moire_img.repeat(1, 3, 1, 1)
            clean_img = clean_img.repeat(1, 3, 1, 1)
        moire_img = (moire_img-self.mean) / self.std
        clean_img = (clean_img-self.mean) / self.std
        if self.resize:
            moire_img = self.transform(moire_img, mode='bilinear', size=(224, 224), align_corners=False)
            clean_img = self.transform(clean_img, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = moire_img
        y = clean_img
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                if use_fft:
                    x = fft_magnitude(x)
                    y = fft_magnitude(y)
                    
                if self.l1_loss is None:
                    loss += torch.nn.functional.l1_loss(x, y)
                else:
                    loss += self.l1_loss(
                        output=x, clean_img=y,
                        clean_img_freq=0, calc_mean=True, eps=0, eps_2=0,
                    )
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)

                # normalize
                gram_x = gram_x / (gram_x.sum() + 1e-6)
                gram_y = gram_y / (gram_y.sum() + 1e-6)

                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
                
        return loss
    

def img_pad(x, h_r=0, w_r=0):
    '''
    Here the padding values are determined by the average r,g,b values across the training set
    in FHDMi dataset. For the evaluation on the UHDM, you can also try the commented lines where
    the mean values are calculated from UHDM training set, yielding similar performance.
    '''
    x1 = F.pad(x[:, 0:1, ...], (w_r, w_r, h_r, h_r), value=0.3827)
    x2 = F.pad(x[:, 1:2, ...], (w_r, w_r, h_r, h_r), value=0.4141)
    x3 = F.pad(x[:, 2:3, ...], (w_r, w_r, h_r, h_r), value=0.3912)
    
    y = torch.cat([x1, x2, x3], dim=1)
    
    return y


def get_time_diff(start_time, end_time):
    time_diff = end_time - start_time
    time_diff = time_diff - datetime.timedelta(microseconds=time_diff.microseconds)
    time_diff = str(time_diff)
    time_diff = time_diff.replace(", ", "+")
    time_diff = time_diff.replace(" ", "")

    _, current_time = get_current_time()

    return current_time + "|" + time_diff


def get_current_time():
    datetime_now = datetime.datetime.now()
    datetime_now_1 = str(datetime_now)
    datetime_now_1 = datetime_now_1.replace(":", ".")
    datetime_now_1 = datetime_now_1.replace(" ", "]_[")

    datetime_now_2 = datetime_now
    datetime_now_2 = datetime_now_2.replace(microsecond=0)
    datetime_now_2 = str(datetime_now_2)
    datetime_now_2 = datetime_now_2.replace(" ", "_")
    datetime_now_2 = datetime_now_2.replace("-", "")
    datetime_now_2 = datetime_now_2.replace(":", "")
    # year with 2 digits
    datetime_now_2 = datetime_now_2[2:]
    
    return datetime_now_1, datetime_now_2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MoireDataset(Dataset):
    """Moire dataset with OpenCV/Image PIL."""

    def __init__(self, data_path, sub_dir, moire_dir, clean_dir, data_name, is_training, patch_size=None, try_it=False):
        """
        This function returns a dataset object.
        
        :param data_path: the path to the data folder
        :param sub_dir: the directory of the data set
        :param moire_dir: the directory where the moire images are stored
        :param clean_dir: the directory of the clean images
        :param patch_size: the size of the patches to be extracted
        """
        if data_name == "uhdm":
            if is_training:
                self.moire_list = []
                self.clean_list = []
                for dir_path in natsorted(glob(os.path.join(data_path, sub_dir, "*"))):
                    for moire_fn in natsorted(glob(os.path.join(data_path, sub_dir, dir_path, "*_moire.jpg"))):
                        self.moire_list.append(moire_fn)
                        self.clean_list.append(moire_fn.replace("_moire.jpg", "_gt.jpg"))
            else:
                self.moire_list = natsorted(glob(os.path.join(data_path, sub_dir, "*_moire.jpg")))
                self.clean_list = natsorted(glob(os.path.join(data_path, sub_dir, "*_gt.jpg")))
        else:
            self.moire_list = natsorted(glob(os.path.join(data_path, sub_dir, moire_dir, "*")))
            self.clean_list = natsorted(glob(os.path.join(data_path, sub_dir, clean_dir, "*")))
        
        # if try_it:
        #     self.moire_list = self.moire_list[:20]
        #     self.clean_list = self.clean_list[:20]

        self.data_name = data_name
        self.is_training = is_training
        self.patch_size = patch_size

    def __len__(self):
        return len(self.moire_list)

    def __getitem__(self, idx):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if torch.is_tensor(idx):
            idx = idx.tolist()

        moire_fn = self.moire_list[idx]
        clean_fn = self.clean_list[idx]

        moire_img = Image.open(moire_fn).convert('RGB')
        clean_img = Image.open(clean_fn).convert('RGB')

        if self.data_name == "uhdm":
            if self.is_training:
                if self.patch_size:
                    img_size = self.patch_size
                else:
                    img_size = 1536

                if os.path.split(clean_fn)[0][-5:-3] == 'mi':
                    w = 4624
                    h = 3472
                else:
                    w = 4032
                    h = 3024
                xxx = random.randint(0, w - img_size)
                yyy = random.randint(0, h - img_size)

                moire_img = moire_img.crop((xxx, yyy, xxx + img_size, yyy + img_size))
                clean_img = clean_img.crop((xxx, yyy, xxx + img_size, yyy + img_size))
            else:
                if self.patch_size:
                    img_size = self.patch_size

                    w, h = moire_img.size
                    
                    # center crop
                    xxx = int((w - img_size) / 2)
                    yyy = int((h - img_size) / 2)

                    moire_img = moire_img.crop((xxx, yyy, xxx + img_size, yyy + img_size))
                    clean_img = clean_img.crop((xxx, yyy, xxx + img_size, yyy + img_size))

        elif self.data_name == "fhdmi":
            if self.is_training:
                if self.patch_size:
                    img_size = self.patch_size
                else:
                    img_size = 1024
                                    
                xxx = random.randint(0, 1920 - img_size)
                yyy = random.randint(0, 1080 - img_size)

                moire_img = moire_img.crop((xxx, yyy, xxx + img_size, yyy + img_size))
                clean_img = clean_img.crop((xxx, yyy, xxx + img_size, yyy + img_size))

        elif self.data_name == "tip18":
            w, h = moire_img.size
            xxx = 0
            yyy = 0
            if self.is_training:
                xxx = random.randint(-6, 6)
                yyy = random.randint(-6, 6)

            moire_img = moire_img.crop((int(w / 6) + xxx, int(h / 6) + yyy, int(w * 5 / 6) + xxx, int(h * 5 / 6) + yyy))
            clean_img = clean_img.crop((int(w / 6) + xxx, int(h / 6) + yyy, int(w * 5 / 6) + xxx, int(h * 5 / 6) + yyy))

            moire_img = moire_img.resize((256, 256), Image.BILINEAR)
            clean_img = clean_img.resize((256, 256), Image.BILINEAR)

        elif self.data_name == "aim":
            if self.is_training:
                if self.patch_size:
                    img_size = self.patch_size
                else:
                    img_size = 512

                xxx = random.randint(0, 1024 - img_size)
                yyy = random.randint(0, 1024 - img_size)

                moire_img = moire_img.crop((xxx, yyy, xxx + img_size, yyy + img_size))
                clean_img = clean_img.crop((xxx, yyy, xxx + img_size, yyy + img_size))

        # images to tensors
        moire_img = transforms.ToTensor()(moire_img)
        clean_img = transforms.ToTensor()(clean_img)

        # return sample
        sample = {
            "moire": moire_img, "clean": clean_img,
            "moire_fn": moire_fn, "clean_fn": clean_fn
        }

        return sample


def get_tensor_value(x):
    # check if x is a GPU tensor
    try:
        if x.is_cuda:
            return x.cpu()
        else:
            return x
    except:
        return x


def log_system_info():
    try:
        """
        Logs system information, PyTorch environment, and installed packages
        to stdout only.
        """
        lines = []
        lines.append("="*60)
        lines.append(f"Run started: {datetime.datetime.now().isoformat()}")
        lines.append("="*60)

        # Basic system info
        lines.append("\n==== System Info ====")
        lines.append(f"Platform: {platform.platform()}")
        lines.append(f"Hostname: {platform.node()}")
        lines.append(f"Machine : {platform.machine()}")
        lines.append(f"Processor: {platform.processor()}")
        lines.append(f"CPU count: {os.cpu_count()}")
        lines.append(f"Memory  : {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")
        lines.append(f"Swap    : {psutil.swap_memory().total / (1024 ** 3):.2f} GB")
        lines.append(f"Is this a Docker container? {'Yes' if os.path.exists('/.dockerenv') else 'No'}")

        # Python info
        lines.append("\n==== Python Info ====")
        lines.append(f"Python   : {platform.python_version()} ({sys.executable})")
        lines.append(f"Compiler : {platform.python_compiler()}")
        lines.append(f"Build    : {platform.python_build()}")

        # PyTorch info
        lines.append("\n==== PyTorch Info ====")
        lines.append(f"Torch    : {torch.__version__}")
        lines.append(f"CUDA avail: {torch.cuda.is_available()}")
        lines.append(f"CUDA ver : {torch.version.cuda}")
        lines.append(f"cuDNN ver: {torch.backends.cudnn.version()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                lines.append(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        # PyTorch detailed collect_env
        lines.append("\n==== PyTorch collect_env ====")
        lines.append(get_pretty_env_info())

        # Installed packages
        lines.append("\n==== Installed Packages (pip freeze) ====")
        try:
            pkgs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
            lines.extend(pkgs.strip().splitlines())
        except Exception:
            lines.append("Failed to run pip freeze, using pkg_resources fallback.")
            for dist in sorted(pkg_resources.working_set, key=lambda x: x.project_name.lower()):
                lines.append(f"{dist.project_name}=={dist.version}")
    except Exception as e:
        lines.append(f"Error logging system info: {str(e)}")
        pass

    # Print to stdout
    print("\n".join(lines))


def get_grad_vector(grads):
    vec = []
    for g in grads:
        if g is not None:
            vec.append(g.reshape(-1))
    return torch.cat(vec)


def tcsvt_contrastive_loss(criterion, C_1, C_2, C_3, P_1, P_2, P_3, M_1, args):
    # anchor to positive distance
    C2P_dist = (
        criterion(C_1, P_1) + \
            criterion(C_2, P_2) + \
                criterion(C_3, P_3)
    )
    
    with torch.no_grad():
        # ---------------------------------------------------
        # negative sampling
        # ---------------------------------------------------
        lambda_P = random.uniform(0, 1) + 0.5
        NP_1 = torch.clamp(P_1 + lambda_P * M_1.detach(), 0, 1)
        NP_2 = F.interpolate(NP_1, scale_factor=0.5, mode='bilinear', align_corners=False)
        NP_3 = F.interpolate(NP_1, scale_factor=0.25, mode='bilinear', align_corners=False)
        NP_1 = NP_1.detach()
        NP_2 = NP_2.detach()
        NP_3 = NP_3.detach()
        
        lambda_C = random.uniform(0, 1) + 0.5
        NC_1 = torch.clamp(C_1.detach() + lambda_C * M_1.detach(), 0, 1)
        NC_2 = F.interpolate(NC_1, scale_factor=0.5, mode='bilinear', align_corners=False)
        NC_3 = F.interpolate(NC_1, scale_factor=0.25, mode='bilinear', align_corners=False)
        NC_1 = NC_1.detach()
        NC_2 = NC_2.detach()
        NC_3 = NC_3.detach()

        # weights
        NP2P_dist_nograd = criterion(NP_1, P_1) + \
            criterion(NP_2, P_2) + \
                criterion(NP_3, P_3)
        NC2P_dist_nograd = criterion(NC_1, P_1) + \
            criterion(NC_2, P_2) + \
                criterion(NC_3, P_3)
        C2P_dist_nograd = C2P_dist.detach()
    
    # anchor to negative (P) distance
    C2NP_dist = (
        criterion(C_1, NP_1.detach()) + \
            criterion(C_2, NP_2.detach()) + \
                criterion(C_3, NP_3.detach())
    )
    
    # anchor to negative (C) distance
    C2NC_dist = (
        criterion(C_1, NC_1.detach()) + \
            criterion(C_2, NC_2.detach()) + \
                criterion(C_3, NC_3.detach())
    )
    
    # ---------------------------------------------------
    # contrastive loss computation
    # ---------------------------------------------------
    eps = 1e-6

    NP_weight = torch.clamp(NP2P_dist_nograd / (C2P_dist_nograd + eps), 0, 10)
    NC_weight = torch.clamp(NC2P_dist_nograd / (C2P_dist_nograd + eps), 0, 10)

    den = C2P_dist + NP_weight * C2NP_dist + NC_weight * C2NC_dist
    prob = C2P_dist / (den + eps)
    
    contloss_num = args.losses.contrastive.weight * (
        -torch.log(prob + eps)
    )
    
    return contloss_num
