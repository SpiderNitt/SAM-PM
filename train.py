import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelSummary, LearningRateMonitor
from segment_anything import sam_model_registry
from segment_anything.modeling.camosam import CamoSam

import wandb

from dataloaders.camo_dataset import get_loader
from dataloaders.vos_dataset import get_loader as get_loader_moca
from dataloaders.moca_test import get_test_loader
from callbacks import WandB_Logger
from config import cfg

L.seed_everything(2023, workers=True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('highest')
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)

ckpt = None

if cfg.model.propagation_ckpt:
    ckpt = torch.load(cfg.model.propagation_ckpt)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = sam_model_registry[cfg.model.type](checkpoint=cfg.model.checkpoint, cfg=cfg)
model = CamoSam(cfg, model, ckpt=ckpt)

wandblogger = WandbLogger(project="CVPR_Final", save_code=True, settings=wandb.Settings(code_dir="."))

# torch._dynamo.config.verbose=True # for debugging
lr_monitor = LearningRateMonitor(logging_interval='epoch')
model_weight_callback = WandB_Logger(cfg, wandblogger)

callbacks = [ModelSummary(max_depth=3), lr_monitor, model_weight_callback]

trainer = L.Trainer(
    accelerator=device,
    devices=cfg.num_devices,
    callbacks=callbacks,
    precision=cfg.precision,
    logger=wandblogger,
    max_epochs=cfg.num_epochs,
    num_sanity_val_steps=0,
    # strategy="ddp",
    log_every_n_steps=15,
    enable_checkpointing=True,
    profiler='simple',
    # overfit_batches=1
)
if trainer.global_rank == 0:
    wandblogger.experiment.config.update(dict(cfg))

if cfg.dataset.stage1:
    train_dataloader = get_loader_moca(cfg.dataset)
else:
    train_dataloader = get_loader(cfg.dataset)
trainer.fit(model, train_dataloader)