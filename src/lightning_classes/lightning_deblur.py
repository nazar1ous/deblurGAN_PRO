from collections import OrderedDict
import pytorch_lightning as pl
from kornia.utils import tensor_to_image
from pytorch_lightning.metrics.regression import PSNR, SSIM
from torch import cat, stack
from torchvision.utils import make_grid
import wandb
from src.utils.dataset import *
from torch.utils.data import random_split
from src.models.get_models import get_generator, get_discriminator
from omegaconf import DictConfig
# from src.models.SRGAN import Loss
# from src.losses.wgan_gp import DiscLossWGANGP
from src.losses.ragan_ls import RelativisticDiscLossLS
from src.losses.get_losses import l1_loss, fft_loss, get_adv_loss_module


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class LightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.generator = get_generator(self.cfg.generator.name)
        self.discriminator = get_discriminator(self.cfg.discriminator.name)
        self.dataset_path = self.cfg.dataset.folder_path
        self.batch_size = self.cfg.data.batch_size
        self._device = self.cfg.data.device

        # cache for generated images
        self.last_source_imgs = None
        self.last_generated_imgs = None
        self.last_gt_target_imgs = None
        self._usewandb = self.cfg.wandb.use_wandb

        self._cur_train_epoch = 0
        self._cur_valid_epoch = 0
        self._train_step = 0
        self._valid_step = 0
        self.generated = None

        self.psnr_metric = PSNR(data_range=2.0)
        self.ssim_metric = SSIM(data_range=2.0)

        self._loss = get_adv_loss_module(self.cfg.adv_loss.name)
        self.adv_loss = self._loss.get_g_loss
        self.disc_loss = self._loss.get_loss

        self.adv_loss_lambda = self.cfg.adv_loss.loss_lambda
        self.l1_loss_lambda = self.cfg.generator_basic_loss.l1.loss_lambda
        # TODO will be used in the future
        self.fft_loss_lambda = self.cfg.generator_basic_loss.fft_loss.loss_lambda


        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.adv_beta = self.cfg.adv_loss.loss_lambda

    def forward(self, source):
        out = self.generator(source)
        return out

    def setup(self, stage):
        if stage == "fit":
            train_dataset = get_train_dataset(self.dataset_path, use_transform=True)
            valid_space_len = int(self.cfg.dataset.valid_percentage * len(train_dataset))

            self.train_dataset, self.valid_dataset = random_split(train_dataset,
                                                                  [len(train_dataset) - valid_space_len, valid_space_len])
        elif stage == "test":
            self.test_dataset = get_test_dataset(self.dataset_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
         num_workers=self.cfg.data.num_workers
        )

    def training_step(self, batch, batch_nb, optimizer_idx):

        if not self.generator.training:
            self.generator.train()

        if not self.discriminator.training:
            self.discriminator.train()

        # train discriminator
        if optimizer_idx == 0:
            source, target = batch
            source = source.to(self._device)
            target = target.to(self._device)

            self.last_source_imgs = source
            self.last_gt_target_imgs = target
            self.generated = self(source=source)
            self.generated.requires_grad = True
            pass_to_disc = self.generated.clone()
            d_loss = self.disc_loss(net=self.discriminator, fakeB=pass_to_disc, realB=self.last_gt_target_imgs)

            tqdm_dict = {'d_loss_train_step': d_loss.detach().clone(),
                         "batch_step": self.global_step * self.batch_size}
            output = OrderedDict({
                'loss': d_loss,
                'log': tqdm_dict
            })
            return output

        # train generator
        if optimizer_idx == 1:
            adv_loss = self.adv_loss_lambda * self.adv_loss(self.discriminator, self.generated, self.last_gt_target_imgs)
            l1_loss_ = self.l1_loss_lambda * l1_loss(self.generated, self.last_gt_target_imgs)
            g_loss = adv_loss + l1_loss_

            generated = self.generated.detach().clone()
            target = self.last_gt_target_imgs.detach().clone()
            ssim_value = self.ssim_metric(pred=generated,
                                          target=target)
            psnr_value = self.psnr_metric(pred=generated,
                                          target=target)

            tqdm_dict = {'g_loss_train_step': g_loss.detach().clone(),
                         'adv_loss_train_step': adv_loss.detach().clone(),
                         'l1_loss_train_step': l1_loss_.detach().clone(),
                         "ssim_value_train_step": ssim_value.detach().clone(),
                         "psnr_value_train_step": psnr_value.detach().clone(),
                         "batch_step": self.global_step * self.batch_size
                         }
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

            return output

    def training_epoch_end(self, outputs):
        img_for_log = None
        try:
            concat_imgs = []
            for i in range(0, self.batch_size):
                concat_imgs.append(cat((
                    self.last_source_imgs[i],
                    self.generated[i],
                    self.last_gt_target_imgs[i],
                ), 2))
            sample_imgs = stack(concat_imgs)
            sample_imgs = (sample_imgs + 1) / 2.0  # [-1; 1] -> [0; 1]

            grid = make_grid(sample_imgs, nrow=1)
            img_for_log = tensor_to_image(grid)


            self.logger.experiment.add_image(f"on_train_epoch_end", img_for_log, self.current_epoch,
                                             dataformats="HWC")

        except Exception as e:
            print("can't log images in training epoch end")
            print(str(e))

        avg_g_loss = torch.stack([x['log']['g_loss_train_step'] for x in outputs[1]]).mean()
        avg_l1_loss = torch.stack([x['log']['l1_loss_train_step'] for x in outputs[1]]).mean()
        avg_adv_loss = torch.stack([x['log']['adv_loss_train_step'] for x in outputs[1]]).mean()
        avg_d_loss = torch.stack([x['log']['d_loss_train_step'] for x in outputs[0]]).mean()
        avg_ssim_value = torch.stack([x['log']['ssim_value_train_step'] for x in outputs[1]]).mean()
        avg_psnr_value = torch.stack([x['log']['psnr_value_train_step'] for x in outputs[1]]).mean()

        tqdm_dict = {'avg_train_g_loss': avg_g_loss,
                     'avg_train_d_loss': avg_d_loss,
                     'avg_train_adv_loss': avg_adv_loss,
                     'avg_train_l1_loss': avg_l1_loss,
                     "avg_train_ssim_value": avg_ssim_value,
                     "avg_train_psnr_value": avg_psnr_value,
                     'epoch': self.current_epoch,
                     'G_lr': get_lr(self.trainer.optimizers[1]),
                     'D_lr': get_lr(self.trainer.optimizers[0]),
                     }

        dct = {'train/' + key: value for key, value in tqdm_dict.items()}
        if self._usewandb:
            if img_for_log is not None:
                dct.update({'train/img': wandb.Image(img_for_log,
                                                     caption=f'{self._cur_train_epoch}')})

            wandb.log(dct)

        output = OrderedDict({
            'avg_train_g_loss': avg_g_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        self._cur_train_epoch += 1
        return output

    def validation_step(self, batch, batch_idx):

        if self.generator.training:
            self.generator.eval()

        if self.discriminator.training:
            self.discriminator.eval()
        source, target = batch
        source = source.to(self._device)
        target = target.to(self._device)
        generated = self(source=source)

        self.val_generated = generated
        self.val_source = source
        self.val_target = target
        fake_det = self.val_generated.detach().clone()
        gt_det = self.val_target.detach().clone()
        ssim_value = self.ssim_metric(pred=fake_det,
                                      target=gt_det)
        psnr_value = self.psnr_metric(pred=fake_det,
                                      target=gt_det)

        tqdm_dict = {
                     "ssim_value_val_step": ssim_value,
                     "psnr_value_val_step": psnr_value,
                     "batch_step_val_step": self.global_step * self.batch_size
                     }
        output = {
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

        return output

    def validation_epoch_end(self, outputs):
        img_for_log = None
        try:
            concat_imgs = []
            for i in range(0, self.batch_size):
                concat_imgs.append(cat((
                    self.val_source[i],
                    self.val_generated[i],
                    self.val_target[i],
                ), 2))

            sample_imgs = stack(concat_imgs)
            sample_imgs = (sample_imgs + 1) / 2.0  # [-1; 1] -> [0; 1]
            grid = make_grid(sample_imgs, nrow=1)
            img_for_log = tensor_to_image(grid)

            self.logger.experiment.add_image(f"on_val_epoch_end", img_for_log, self.current_epoch,
                                             dataformats="HWC")

        except Exception as e:
            print("can't log images in validation epoch end")
            print(str(e))

        avg_ssim_value = torch.stack([x['log']['ssim_value_val_step'] for x in outputs]).mean()
        avg_psnr_value = torch.stack([x['log']['psnr_value_val_step'] for x in outputs]).mean()

        tqdm_dict = {
                     "avg_val_ssim_value": avg_ssim_value,
                     "avg_val_psnr_value": avg_psnr_value,
                     'epoch': self.current_epoch,
                     'G_lr': get_lr(self.trainer.optimizers[1]),
                     'D_lr': get_lr(self.trainer.optimizers[0]),
                     }

        output = OrderedDict({
            "avg_val_psnr_value": avg_psnr_value,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        dct = {'valid/' + key: value for key, value in tqdm_dict.items()}
        if self._usewandb:
            if img_for_log is not None:
                dct.update({'valid/img': wandb.Image(img_for_log, caption=f'{self._cur_valid_epoch}')})
            wandb.log(dct)

        self._cur_valid_epoch += 1
        return output

    def backward(self, trainer, loss, optimizer, optimizer_idx: int) -> None:
        if optimizer_idx == 0:
            loss.backward(retain_graph=True)
        elif optimizer_idx == 1:
            loss.backward()

    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(self.generator.parameters(),
                                               lr=self.cfg.optimizer.Adam.lr,
                                               betas=(
                                                   self.cfg.optimizer.Adam.betas.b1,
                                                   self.cfg.optimizer.Adam.betas.b2
                                               ))
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                   lr=self.cfg.optimizer.Adam.lr,
                                                   betas=(
                                                       self.cfg.optimizer.Adam.betas.b1,
                                                       self.cfg.optimizer.Adam.betas.b2
                                                   ))

        generator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=generator_optimizer,
                                                                      **self.cfg.scheduler.params)
        discriminator_scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=discriminator_optimizer,
                                                                      **self.cfg.scheduler.params)

        return [discriminator_optimizer, generator_optimizer], \
               [
                {"scheduler": discriminator_scheduler,
                 "monitor": "avg_train_d_loss",
                 "interval": "epoch",
                 "reduce_on_plateau": True},
               {"scheduler": generator_scheduler,
                "monitor": "avg_train_g_loss",
                "interval": "epoch",
                "reduce_on_plateau": True
                }
               ]