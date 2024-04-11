import torch
import lightning as L
from segment_anything import sam_model_registry
from segment_anything.modeling.camosam import CamoSam

import argparse

from dataloaders.moca_test import get_test_loader
from config import cfg

L.seed_everything(2023, workers=True)
torch.set_float32_matmul_precision('highest')

parser = argparse.ArgumentParser(description='Testing Code !!!')
parser.add_argument('--ckpt', type=str, default="ckpt/pm.pth")
args = parser.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = sam_model_registry[cfg.model.type](checkpoint=cfg.model.checkpoint, cfg=cfg)
model = CamoSam(cfg, model, ckpt=ckpt)

trainer = L.Trainer(
    accelerator=device,
    devices=cfg.num_devices,
    precision=cfg.precision,
    num_sanity_val_steps=0,
    log_every_n_steps=1,
)

test_dataloader = get_test_loader(dataset="moca")
trainer.test(model, test_dataloader)