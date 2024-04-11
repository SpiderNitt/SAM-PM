import os
from os import path, replace
import random
import torch
from torch.utils.data.dataset import Dataset
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.nn import functional as F

from PIL import Image
import numpy as np
from torchvision.transforms.functional import resize, pil_to_tensor
from segment_anything.utils.transforms import ResizeLongestSide
from dataloaders.range_transform import im_normalization, im_mean
from dataloaders.reseed import reseed

class VOSDataset(Dataset):
    """
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, im_root, gt_root, vid_list, max_jump, subset=None, num_frames=3, max_num_obj=3, test=False, cfg=None):
        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump
        self.test = test
        self.videos = []
        self.frames = {}
        self.cfg = cfg

        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

        # Pre-filtering
        for vid in vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            if cfg.name == "moca":
                frames = sorted(os.listdir(os.path.join(self.im_root, vid, "Imgs")))
            else:
                frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < num_frames:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if test else 15, shear=0 if test else 10, interpolation=InterpolationMode.BILINEAR, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if test else 15, shear=0 if test else 10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])


        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop((1080, 1920), scale=(0.7,1.00), ratio=(1920/1080,1920/1080), interpolation=InterpolationMode.NEAREST)
        ])
        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop((1080, 1920), scale=(0.7,1.00), ratio=(1920/1080,1920/1080), interpolation=InterpolationMode.NEAREST)
        ])

        self.resize_longest = ResizeLongestSide(1024)
        self.resize_longest_mask = ResizeLongestSide(256)
    
    def preprocess_prev_masks(self, x):
        h, w = x.shape[-2:]
        padh = 256 - h
        padw = 256 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        idx = idx%len(self.videos)
        video = self.videos[idx]
        info = {}
        info['name'] = video
        
        if self.cfg.name == "moca":
            vid_im_path = path.join(self.im_root, video, "Imgs")
            vid_gt_path = path.join(self.gt_root, video, "GT")
        else:
            vid_im_path = path.join(self.im_root, video) 
            vid_gt_path = path.join(self.gt_root, video)

        frames = self.frames[video]
        trials = 0
        while trials < 5:
            info['frames'] = [] # Appended with actual frames

            num_frames = self.cfg.num_frames
            length = len(frames)
            this_max_jump = min(len(frames), self.max_jump)

            # iterative sampling
            if self.test:
                frames_idx = [3]
                reseed(2023)
            frames_idx = [np.random.randint(length)]
            acceptable_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1))).difference(set(frames_idx))
            while(len(frames_idx) < num_frames):
                idx = np.random.choice(list(acceptable_set))
                frames_idx.append(idx)
                new_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1)))
                acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))

            frames_idx = sorted(frames_idx)
            if self.test:
                frames_idx = [3, 4, 5]
            if np.random.rand() < 0.5 and not self.test:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_objects = []
            cropped_img = []
            for f_idx in frames_idx:
                jpg_name = frames[f_idx][:-4] + '.jpg'
                png_name = frames[f_idx][:-4] + '.png'
                info['frames'].append(jpg_name[:-4])

                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                if not self.test:
                    reseed(sequence_seed)
                    this_im = self.all_im_dual_transform(this_im)
                    this_im = self.all_im_lone_transform(this_im)
                
                    reseed(sequence_seed)
                    this_gt = self.all_gt_dual_transform(this_gt)

                    pairwise_seed = np.random.randint(2147483647)
                    reseed(pairwise_seed)
                    this_im = self.pair_im_dual_transform(this_im)
                    this_im = self.pair_im_lone_transform(this_im)
                    reseed(pairwise_seed)
                    this_gt = self.pair_gt_dual_transform(this_gt)

                cropped_img.append(pil_to_tensor(this_im))
                this_im = torch.as_tensor(self.resize_longest.apply_image(np.array(this_im, dtype=np.uint8))).permute(2, 0, 1)
                resize_longest_size = this_im.shape[-2:]
                this_im = self.preprocess(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels!=0]

            if len(labels) == 0:
                target_objects = []
                trials += 1
            else:
                target_objects = labels.tolist()
                break
        
        if trials == 5:
            return self.__getitem__(np.random.randint(len(self.videos)))
        
        if len(target_objects) > self.cfg.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.cfg.max_num_obj, replace=False)

        info['num_objects'] = max(1, len(target_objects))

        masks = np.stack(masks, 0)

        H, W = tuple(masks.shape[1:])
        # Generate one-hot ground-truth
        cls_gt = np.zeros((self.cfg.num_frames, H, W), dtype=np.int64)
        all_frame_gt = np.zeros((self.cfg.num_frames, self.cfg.max_num_obj, H, W), dtype=np.int64) # Shape explains itself
        for t in range(self.cfg.num_frames):
            for i, l in enumerate(target_objects):
                this_mask = (masks==l)
                cls_gt[this_mask] = i+1
                all_frame_gt[t,i] = (this_mask[t])

        cls_gt = np.expand_dims(cls_gt, 1)
        all_frame_gt_256 = all_frame_gt.reshape(-1, H, W)
        
        new_all_frame_gt = []
        for t in range(len(all_frame_gt_256)):
            new_all_frame_gt.append(torch.as_tensor(self.resize_longest_mask.apply_image(all_frame_gt_256[t].astype(dtype=np.uint8))))

        new_all_frame_gt = torch.stack(new_all_frame_gt, 0).reshape(-1, self.cfg.max_num_obj, *new_all_frame_gt[0].shape[-2:])
        new_all_frame_gt = self.preprocess_prev_masks(new_all_frame_gt).float()

        new_prev_frame_gt = new_all_frame_gt[:-1]

        all_frame_gt = torch.as_tensor(all_frame_gt).float()

        # 1 if object exist, 0 otherwise
        selector = [1 if i < info['num_objects'] else 0 for i in range(self.cfg.max_num_obj)]
        selector = torch.BoolTensor(selector)

        # (H', W') -> after LongestSideResize
        data = {
            'image': images, # (num_frames, 3, 1024, 1024) 
            'gt_mask': all_frame_gt, # (num_frames, num_obj=3, H, W)
            'gt_mask_256': new_all_frame_gt, # (num_frames, num_obj=3, 256, 256)
            'prev_masks': new_prev_frame_gt, # (num_frames, num_obj=3, 256, 256)
            'selector': selector, # (num_obj=3) Indicates if ith object exists
            'cropped_img': cropped_img, # (num_frames, 3, H, W)
            'original_size': list(all_frame_gt.shape[-2:]),
            'resize_longest_size': list(resize_longest_size),
            'info': info
        }
        return data

    def __len__(self):
        return len(self.videos)
    
def collate_fn(batch):
    output = {}
    for key in batch[0].keys():
        output[key] = [d[key] for d in batch]
        if key in ["image", "prev_masks", "selector"]:
            output[key] = torch.stack(output[key], 0)
    
    return output

def get_loader(cfg):
    train_list = sorted(os.listdir(cfg.root_dir+'MoCA_Video/TrainDataset_per_sq'))
    test_list = sorted(os.listdir(cfg.root_dir+'MoCA_Video/TestDataset_per_sq'))

    print("Training Samples: ",len(train_list))
    train_dataset = VOSDataset(cfg.root_dir+'MoCA_Video/TrainDataset_per_sq', 
                            cfg.root_dir+'MoCA_Video/TrainDataset_per_sq', 
                            train_list ,max_jump=cfg.max_jump, 
                            num_frames=cfg.num_frames,  
                            max_num_obj=cfg.max_num_obj, 
                            test=False,
                            cfg=cfg)
    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        persistent_workers=cfg.persistent_workers,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
        drop_last=True
    )

    return train_data_loader