# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
from metrics import jaccard_dice
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from .modules import Memory

import lightning as L

class CamoSam(L.LightningModule):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        config,
        model,
        ckpt = None
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.
        """
        super().__init__()
        self.ckpt = ckpt
        self.model = model
        self.cfg = config
        if ckpt is not None:
            self.model.propagation_module.load_state_dict(ckpt['model_state_dict'])
            print("!!! Loaded checkpoint for propagation module !!!")

        self.set_requires_grad()

        self.epoch_freq = self.cfg.train_metric_interval
        
        self.train_benchmark = []
        self.val_benchmark = []
        
    def set_requires_grad(self):
        model_grad = self.cfg.model.requires_grad

        for param in self.model.image_encoder.parameters():
            param.requires_grad = model_grad.image_encoder

        for param in self.model.prompt_encoder.parameters():
            param.requires_grad = model_grad.prompt_encoder

        for param in self.model.mask_decoder.parameters():
            param.requires_grad = model_grad.mask_decoder
        
        for param in self.model.propagation_module.parameters():
            param.requires_grad = model_grad.propagation_module

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr= self.cfg.opt.learning_rate if self.cfg.opt.learning_rate else 0,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.cfg.opt.weight_decay,
            amsgrad=False,
        )

        if self.ckpt and not self.cfg.opt.learning_rate:
            try:
                optimizer.load_state_dict(self.ckpt['optimizer_state_dict']) # Try-except to handle the case when only propagation ckpt is to be loaded
                print("!!! Loaded optimizer state dict !!!")
            except:
                pass
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.opt.steps, gamma=self.cfg.opt.decay_factor)
        return [optimizer], [scheduler]

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        
        return self.model(batched_input, multimask_output)
    
    def check_frequency(self, check_idx):
        return check_idx % self.epoch_freq == 0

    @torch.no_grad()
    def log_images(self, information, img_dict, sep_mask_dict, mask_dict, gt_mask_dict, iou_dict, batch_idx, train=True):
        start_idx = 1 if self.cfg.dataset.stage1 else self.cfg.dataset.num_frames-1
        for frame_idx in range(len(information)):
            log_dict = []
            for t in range(start_idx, self.cfg.dataset.num_frames):
                log_dict.append({"ground_truth": {"mask_data" : gt_mask_dict[t][frame_idx].cpu().numpy()}, "prediction": {"mask_data" : mask_dict[t][frame_idx].cpu().numpy()}})
                for obj_index, sep_obj in enumerate(sep_mask_dict[t][frame_idx]):
                    log_dict[-1][f"prediction_{obj_index + 1}"] = {"mask_data" : sep_obj.cpu().numpy() * (obj_index + 1)}

            self.logger.log_image(f"Images/{'train' if train else 'test'}/{information[frame_idx]['name']}", [img_dict[t][frame_idx] for t in range(start_idx, self.cfg.dataset.num_frames)], step=self.global_step, masks=log_dict,
                                caption=[f"Epoch_{self.current_epoch}_IoU_{[str(iou_obj)[:5] for iou_obj in iou_dict[t][frame_idx].cpu().tolist()]}_Frame_{information[frame_idx]['frames']}" for t in range(start_idx, self.cfg.dataset.num_frames)])
   
    def sigmoid_focal_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        p: torch.Tensor,
        alpha: float = 0.25, # optimal based on https://arxiv.org/pdf/1708.02002.pdf
        gamma: float = 2,
    ) -> torch.Tensor:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            inputs (Tensor): A float tensor of arbitrary shape.
                    The predictions for each example.
            targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha (float): Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default: ``0.25``.
            gamma (float): Exponent of the modulating factor (1 - p_t) to
                    balance easy vs hard examples. Default: ``2``.
            reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                    ``'none'``: No reduction will be applied to the output.
                    ``'mean'``: The output will be averaged.
                    ``'sum'``: The output will be summed. Default: ``'none'``.
        Returns:
            Loss tensor with the reduction option applied.
        """
        # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

        # p = torch.sigmoid(inputs)
        inputs = inputs.flatten(-2)	 # [num_true_obj, C, HxW]
        targets = targets.flatten(-2) # [num_true_obj, C, HxW]
        p = p.flatten(-2) # [num_true_obj, C, HxW]
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma) # [C, H, W]
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(dim=-1) # [num_true_obj, C]

    def dice_loss(self, inputs, targets):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        # 2ypˆ+ 1 /(y + ˆp + 1)
  
        # inputs = inputs.sigmoid()
        inputs = inputs.flatten(-2)	 # [num_true_obj, C, HxW]
        targets = targets.flatten(-2) # [num_true_obj, C, HxW]
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss # [num_true_obj, C]
    
    def iou(self, inputs, targets):
        """
        Compute the IOU
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        # ypˆ+ 1 /(y + ˆp - ypˆ+ 1)
  
        # inputs = inputs.sigmoid()
        inputs = inputs.flatten(-2)	 # [num_true_obj, C, HxW]
        targets = targets.flatten(-2) # [num_true_obj, C, HxW]
        numerator = (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1) - numerator
        iou = (numerator + 1) / (denominator + 1)
        return iou # [num_true_obj, C]

    def training_step(self, batch, batch_idx):
        img_embeddings = self.model.getImageEmbeddings(batch['image']) # (B, F=3, 256, 64, 64)
        bs = len(img_embeddings)

        pred_masks_dict = {}
        iou_pred_dict = {}

        # output_0, output_1, prop_pos_embed = self(batch, self.cfg.model.multimask_output)
        loss_focal = 0
        loss_dice = 0
        loss_iou = 0
        loss_total = 0

        low_res_pred_list = torch.empty((bs, 0, self.cfg.dataset.max_num_obj, 256, 256), device=batch['image'].device)
        batch_indexing = torch.arange(self.cfg.dataset.max_num_obj, device=batch['image'].device) # [P]
        total_num_objects = 0

        start_idx = 1 if self.cfg.dataset.stage1 else self.cfg.dataset.num_frames-1
        for t in range(start_idx, self.cfg.dataset.num_frames):
            outputs, prop_log_dict = self.model.getPropEmbeddings(img_embeddings, batch, low_res_pred_list, multimask_output=self.cfg.model.multimask_output, t=t)
            low_res_pred_list_tmp = []
            pred_masks_dict[t] = []
            iou_pred_dict[t] = []

            for each_output, gt_mask, dense_embed, selector, resize_longest_size in zip(outputs, batch['gt_mask_256'], prop_log_dict["prop_dense_embed"], batch['selector'], batch['resize_longest_size']): # selector = [True, True, False]
                total_num_objects += selector.sum()
                pred_masks = each_output["low_res_logits"][..., :resize_longest_size[0]//4, :resize_longest_size[1]//4] # [P=3, C, H, W]
                # pred_masks_list.append(pred_masks.detach())
                gt_mask = gt_mask[t].unsqueeze(1) # [P, 1, H, W]
                gt_mask = gt_mask.repeat((1, pred_masks.shape[1], 1, 1)) # [P, C, H, W] 
                gt_mask = gt_mask[..., :resize_longest_size[0]//4, :resize_longest_size[1]//4]
                
                pred_masks_sigmoid = torch.sigmoid(pred_masks)
                loss_focal_tmp = self.sigmoid_focal_loss(pred_masks, gt_mask, pred_masks_sigmoid)
                loss_dice_tmp = self.dice_loss(pred_masks_sigmoid, gt_mask)
                loss_iou_tmp = F.mse_loss(
                    each_output["iou_predictions"],
                    self.iou(pred_masks_sigmoid, gt_mask),
                    reduction="none",
                )
                loss_tmp = (
                    self.cfg.focal_wt * loss_focal_tmp
                    + loss_dice_tmp
                    + loss_iou_tmp
                ) # [P, C]

                loss_tmp, min_idx = torch.min(loss_tmp, -1) # [P]
                loss_total += loss_tmp[selector].sum() # (num_true_obj)

                loss_focal += loss_focal_tmp[batch_indexing, min_idx][selector].sum()
                loss_dice += loss_dice_tmp[batch_indexing, min_idx][selector].sum()
                loss_iou += loss_iou_tmp[batch_indexing, min_idx][selector].sum() # [num_true_obj]

                low_res_pred_list_tmp.append(each_output["low_res_logits"][batch_indexing, min_idx]) # (P=3, C, 256, 256) -> (P=3, 256, 256)
                pred_masks_dict[t].append(torch.sigmoid(each_output["masks"])[batch_indexing, min_idx][selector].detach())
                iou_pred_dict[t].append(each_output["iou_predictions"][batch_indexing, min_idx][selector].detach())
            
            low_res_pred_list_tmp = torch.stack(low_res_pred_list_tmp).unsqueeze(1) # (B, 1, P=3, 256, 256)
            low_res_pred_list = torch.cat([low_res_pred_list, low_res_pred_list_tmp], dim=1) # (B, t, P=3, 256, 256)

        loss_total = (loss_total) / (total_num_objects)
        avg_focal = (self.cfg.focal_wt * loss_focal) / (total_num_objects)
        avg_dice = loss_dice / (total_num_objects)
        avg_iou = loss_iou / (total_num_objects)
        
        self.train_benchmark.append(loss_total.item())
        log_dict = {"Loss/train/total_loss" : loss_total, "Loss/train/focal_loss" : avg_focal, "Loss/train/dice_loss" : avg_dice, "Loss/train/iou_loss" : avg_iou}

        for key in prop_log_dict:
            log_dict[f'{key}/min'] = prop_log_dict[key].min()
            log_dict[f'{key}/max'] = prop_log_dict[key].max()
            log_dict[f'{key}/mean'] = prop_log_dict[key].mean()
            log_dict[f'{key}/sq_mean'] = (prop_log_dict[key] ** 2).mean()
            log_dict[f'{key}/std'] = prop_log_dict[key].std()

        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)

        return {'loss': loss_total, 'masks': pred_masks_dict, 'iou': iou_pred_dict} # List([num_true_obj, H, W])
    
    def on_train_batch_end(self, output, batch, batch_idx):
        if self.check_frequency(self.current_epoch):
            img_dict = {}
            sep_mask_dict = {}
            mask_dict = {}
            gt_mask_dict = {}
            sep_gt_mask_dict = {}
            metrics_all = {}
            
            start_idx = 1 if self.cfg.dataset.stage1 else self.cfg.dataset.num_frames-1
            for t in range(start_idx, self.cfg.dataset.num_frames):
                output_ = output["masks"][t]
                img_dict[t] = []
                sep_mask_dict[t] = []
                mask_dict[t] = []
                gt_mask_dict[t] = []
                sep_gt_mask_dict[t] = []

                total_jaccard = 0
                total_dice = 0
                total_objects = 0


                for each_output, gt_mask, cropped_img, selector in zip(output_, batch['gt_mask'], batch['cropped_img'], batch['selector']):
                    total_objects += selector.sum()
                    gt_mask = gt_mask[t][selector]
                    sep_mask_dict[t].append(each_output>0.5)
                    sep_gt_mask_dict[t].append(gt_mask.type(torch.int8))

                    j, d = jaccard_dice(each_output>0.5, gt_mask.type(torch.bool))
                    total_jaccard += j
                    total_dice += d

                    max_, max_pos = torch.max(gt_mask, dim=0)
                    gt_mask = ((max_pos+1) * (max_)).type(torch.int8)
                    gt_mask_dict[t].append(gt_mask)

                    max_, max_pos = torch.max(each_output, dim=0)
                    mask = ((max_pos+1) * (max_ > 0.5)).type(torch.int8)
                    mask_dict[t].append(mask)
                    
                    img_dict[t].append(cropped_img[t])
                metrics_all[f'Metrics/train/jaccard_single_obj_{t}'] = total_jaccard / total_objects
                metrics_all[f'Metrics/train/dice_single_obj_{t}'] = total_dice / total_objects

            self.log_dict(metrics_all, on_step=True, on_epoch=True, sync_dist=True)
            
            if batch_idx < 5:
                self.log_images(batch['info'], img_dict, sep_mask_dict, mask_dict, gt_mask_dict, output['iou'], batch_idx=batch_idx, train=True)

    def test_step(self, batch, batch_idx):
        memory = Memory(length=self.cfg.dataset.num_frames-1)
        pred_masks_list = []
        if not self.cfg.result_dir:
            result_dir = f'results/{self.logger.version}/{self.current_epoch}/' + batch['info']
        else:
            result_dir = self.cfg.result_dir + '/' + batch['info']
        embeddings_dir = os.path.join('embeddings', batch['info'])
        os.makedirs(result_dir,exist_ok=True)
        os.makedirs(embeddings_dir, exist_ok=True)

        # Define a function to handle embedding fetching or saving
        def get_or_save_embeddings(img, idx):
            embedding_path = os.path.join(embeddings_dir, f'{idx}.pt')
            if os.path.exists(embedding_path):
                embeddings = torch.load(embedding_path)
            else:
                embeddings = self.model.getImageEmbeddings(img.unsqueeze(0)).squeeze(1)
                torch.save(embeddings, embedding_path)
            return embeddings

        first_gt = batch['first_gt']
        first_embed = get_or_save_embeddings(batch['image'][0], batch['frame_num'][0]).squeeze()
        memory.add(first_embed, first_gt, 1)

        gt = np.asarray(Image.open(batch['first_gt_path']), dtype=np.int32)
        gt[gt != 0] = 1
        pred_masks_list.append(gt)
        masks = Image.fromarray(gt * 255).convert("P")
        masks.save(result_dir + '/' + f'{batch["frame_num"][0]}.png')

        for img, frame_num in zip(batch['image'][1:], batch['frame_num'][1:]):
            current_frame_embeddings = get_or_save_embeddings(img, frame_num)
            prev_masks = memory.get_prev_mask()
            prev_masks = prev_masks.view(-1, 1, *prev_masks.shape[-2:])

            prev_frames_embeddings = memory.get_embed().unsqueeze(0)

            masks, low_res_masks, max_iou = self.model.getTestPropEmbeddings(batch, current_frame_embeddings, prev_frames_embeddings, prev_masks, multimask_output=self.cfg.model.multimask_output)
            masks = masks.detach().cpu()
            max_, max_pos = torch.max(masks, dim=0)
            masks = ((max_pos+1) * (max_ > 0)).type(torch.int8)

            masks = masks.cpu().numpy().astype(np.int32)
            pred_masks_list.append(masks)

            masks[masks != 0] = 1
            masks = Image.fromarray(masks*255).convert("P")
            # masks.putpalette(batch['palette'])
            masks.save(result_dir + '/' f'{frame_num}.png')

            low_res_masks = (low_res_masks > 0).squeeze(1).to(dtype=low_res_masks.dtype)

            memory.add(current_frame_embeddings.squeeze(), low_res_masks, max_iou.mean().item())
        return pred_masks_list