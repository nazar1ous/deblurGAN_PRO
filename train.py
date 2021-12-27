from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from src.lightning_classes.lightning_deblur_SRGAN import LightningModule
import wandb


SAVE_PATH = "model_weights/"

if __name__ == "__main__":
    seed_everything(40)
    # warnings.filterwarnings("ignore")

    model = LightningModule()
    model_checkpoint_callback = ModelCheckpoint(
        monitor="avg_val_psnr_value",
        save_top_k=4,
        filepath=SAVE_PATH+"{epoch}-{avg_val_ssim_value:.3f}-{avg_val_psnr_value:.3f}",
        save_last=True,
        period=10
    )

    trainer = Trainer(
        checkpoint_callback=model_checkpoint_callback,
        gpus=0,
        max_epochs=150,
        precision=32,
        weights_summary="full"
    )

    model.train()
    model.cuda()
    wandb.init(project="deblur-UCU-GAN", entity="nazar1ous")
    wandb.run.name = "First run"
    trainer.fit(model)
