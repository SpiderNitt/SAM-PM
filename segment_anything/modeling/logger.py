from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, batch_frequency=5):
        super().__init__()
        self.batch_freq = batch_frequency

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.check_frequency(batch_idx):
            pl_module.log_images(batch, outputs)
    
    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        pl_module.log_images(batch, outputs, train=False)
