# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .modules import PropagationModule


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        cfg = None
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.cfg = cfg
        self.max_num_obj = cfg.dataset.max_num_obj
        self.num_frames = cfg.dataset.num_frames
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        self.propagation_module = PropagationModule(cfg)

        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
    
    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    def getImageEmbeddings(self, input_images):
        self.image_encoder.eval()
        with torch.no_grad():
            image_embeddings = self.image_encoder(input_images.reshape(-1, 3, 1024, 1024)).reshape(len(input_images), -1, 256, 64, 64)  # Output -> (B, F=3, 256, 64, 64)
        return image_embeddings
    
    def getPropEmbeddings(self, image_embeddings, batched_input, low_res_pred, multimask_output=True, t=1):
        prev_masks = batched_input["prev_masks"][:, :(1 if self.cfg.dataset.stage1 else (self.cfg.dataset.num_frames-1))] # (B, F=2/1, P=3, 256, 256)
        low_res_pred = (low_res_pred > self.mask_threshold).to(dtype=low_res_pred.dtype) # (B, t-1, P=3, 256, 256)
        prev_masks = torch.cat([prev_masks, low_res_pred], dim=1) # (B, F+t-1, P=3, 256, 256)

        prev_masks = prev_masks.view(-1, 1, *prev_masks.shape[-2:])
        _, mask_embeddings = self.prompt_encoder(points=None, boxes=None, masks=prev_masks)
        mask_embeddings = mask_embeddings.view(len(batched_input["selector"]), -1, self.max_num_obj, 256, 64, 64) # (B, [F-1]=2, P=3, 256, 64, 64)

        pos_embed = self.prompt_encoder.get_dense_pe() # (256, 64, 64)
        # embeddings = {"current_frame_embeddings": current_frame_embeddings, "prev_frames_embeddings": prev_frames_embeddings, "mask_embeddings": mask_embeddings}
        embeddings = {"current_frame_embeddings": image_embeddings[:, t], "prev_frames_embeddings": image_embeddings[:, :t], "mask_embeddings": mask_embeddings}
        
        all_sparse_embeddings, all_dense_embeddings, log_dict = self.propagation_module(
            embeddings, pos_embed
        )  # (B, P=3, 64, 64, 256)
        all_dense_embeddings = all_dense_embeddings.permute(0, 1, 4, 2, 3) # (B, P=3, 256, 64, 64)

        outputs = []
        for i, (curr_embedding, prop_sparse_embeddings, prop_dense_embeddings) in enumerate(zip(image_embeddings[:, t], all_sparse_embeddings, all_dense_embeddings)):
            # curr_embedding: (256, 64, 64) -> current target frame embedding
            # prop_dense_embeddings: (3, 256, 64, 64) -> basically we have 3 prompts
            # prop_sparse_embeddings: (3, 8, 256) -> basically we have 3 prompts, each prompt has 8 points
            
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=pos_embed,
                sparse_prompt_embeddings=prop_sparse_embeddings,
                dense_prompt_embeddings=prop_dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=list(batched_input["resize_longest_size"][i]),
                original_size=list(batched_input["original_size"][i])
            )
            outputs.append(
                {
                    "masks": masks, # (P=3, C, H, W)
                    "iou_predictions": iou_predictions, # (P=3, C)
                    "low_res_logits": low_res_masks, # (P=3, C, 256, 256)
                }
            )
        log_dict["prop_dense_embed"] = all_dense_embeddings
        # log_dict["prop_sparse_embed"] = all_sparse_embeddings
            
        return outputs, log_dict
    
    def getTestPropEmbeddings(self, batched_input, current_frame_embeddings, prev_frames_embeddings, prev_masks, multimask_output=True):
        _, mask_embeddings = self.prompt_encoder(points=None, boxes=None, masks=prev_masks)
        mask_embeddings = mask_embeddings.view(1, -1, batched_input['num_obj'], 256, 64, 64) # (1, F, P, 256, 64, 64)
        embeddings = {"current_frame_embeddings": current_frame_embeddings, "prev_frames_embeddings": prev_frames_embeddings, "mask_embeddings": mask_embeddings}
        pos_embed = self.prompt_encoder.get_dense_pe()

        all_sparse_embeddings, all_dense_embeddings, _ = self.propagation_module(embeddings, pos_embed)  # (1, P, 64, 64, 256)
        all_dense_embeddings = all_dense_embeddings.permute(0, 1, 4, 2, 3) # (1, P, 256, 64, 64)

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=current_frame_embeddings,
            image_pe=pos_embed,
            sparse_prompt_embeddings=all_sparse_embeddings[0],
            dense_prompt_embeddings=all_dense_embeddings[0],
            multimask_output=multimask_output,
        )

        max_iou, max_index = torch.max(iou_predictions, -1)
        batch_indexing = torch.arange(len(max_index), device=max_index.device)

        masks = self.postprocess_masks(
            low_res_masks,
            input_size=batched_input['resize_longest_size'],
            original_size=batched_input['original_size']
        )

        low_res_masks = low_res_masks[batch_indexing, max_index] # (P, 256, 256)
        masks = masks[batch_indexing, max_index] # (P, 256, 256)

        return masks, low_res_masks, max_iou
    
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks
    
    def train(self, mode: bool = False):
        if not self.cfg.model.requires_grad.image_encoder:
            self.image_encoder.train(False)
        else:
            self.image_encoder.train(mode)
            
        if not self.cfg.model.requires_grad.prompt_encoder:
            self.prompt_encoder.train(False)
        else:
            self.prompt_encoder.train(mode)

        if not self.cfg.model.requires_grad.mask_decoder:
            self.mask_decoder.train(False)
        else:
            self.mask_decoder.train(mode)
            
        if not self.cfg.model.requires_grad.propagation_module:
            self.propagation_module.train(False)
        else:
            self.propagation_module.train(mode)
        
        return self