from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from src.lightning_classes.lightning_deblur import LightningModule
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

CONFIG_PATH = "config"


@hydra.main(config_path=CONFIG_PATH, config_name="config")
def run_train(cfg : DictConfig) -> None:
    # Look at the config before training
    print(OmegaConf.to_yaml(cfg))

    model = LightningModule(cfg=cfg)
    model_checkpoint_callback = ModelCheckpoint(**cfg.model_checkpoint_save.params)

    trainer = Trainer(
        checkpoint_callback=model_checkpoint_callback,
        **cfg.trainer.params
    )

    model.train()
    model.cuda()
    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity)
    wandb.run.name = cfg.wandb.run_name
    trainer.fit(model)

if __name__ == "__main__":
    # seed_everything(40)
    # warnings.filterwarnings("ignore")
    run_train()

