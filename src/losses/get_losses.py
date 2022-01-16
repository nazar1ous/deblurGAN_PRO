import torch
from src.losses.ragan_ls import RelativisticDiscLossLS
from src.losses.wgan_gp import DiscLossWGANGP


def l1_loss(generated, gt_target):
    l1 = torch.nn.L1Loss()
    return l1(generated, gt_target)


def fft_loss(generated, gt_target):
    label_fft = torch.fft.rfft2(gt_target, norm="backward")
    pred_fft = torch.fft.rfft2(generated, norm="backward")
    return l1_loss(pred_fft, label_fft)


def get_adv_loss_module(name):
    if name == "ragan_ls":
        return RelativisticDiscLossLS()
    elif name == "wgan_gp":
        return DiscLossWGANGP()


