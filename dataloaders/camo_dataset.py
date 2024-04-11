import os
from PIL import Image
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.transforms import InterpolationMode
from dataloaders.reseed import reseed
from torchvision.transforms.functional import resize, pil_to_tensor
from dataloaders.range_transform import im_normalization, im_mean
import torch.nn.functional as F

class VideoDataset(data.Dataset):
    def __init__(
        self, cfg, train
    ):
        self.dataset_root = cfg.root_dir
        self.train = train
        self.cfg = cfg
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.image_list = []
        self.gt_list = []
        ####### CAD10K-V3
        
        path = os.path.join(self.dataset_root, "COD10K-v3") 
        
        split_path = os.path.join(path, "Test")
        info = open(split_path+"/CAM-NonCAM_Instance_Test.txt", 'r').readlines()
        
        if self.train:
            split_path = os.path.join(path, "Train")
            info = open(split_path+"/CAM-NonCAM_Instance_Train.txt", 'r').readlines()

        img_path = os.path.join(split_path, "Image")
        gt_path = os.path.join(split_path, "GT_Instance")

        
        for i in info:
            if "[INFO]" in i:
                i = i.lstrip().rstrip()
                if int(i[-1]) >= 1:
                    name = i.split(" ")[-2].split(".")[0]
                    self.image_list.append(os.path.join(img_path, name+'.jpg'))
                    self.gt_list.append(os.path.join(gt_path, name+'.png'))
                  
        count = len(self.image_list)
        print(f'{"Train" if self.train else "Test"} Loaded COD10K', count)
        assert len(self.image_list) == len(self.gt_list)
          
        
        print(f'===={"Train" if self.train else "Test"} Loaded {count} images')

        # self.pair_im_lone_transform = transforms.Compose([
        #     # transforms.ColorJitter(0.1, 0.05, 0.05, 0), # No hue change here as that's not realistic
        # ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=(0, 15) if train else 0, shear=(0, 10) if train else 0, translate=(0.2, 0.2) if train else None, scale=(0.2, 0.7) if train else None, interpolation=InterpolationMode.BILINEAR, fill=im_mean),
            transforms.GaussianBlur(kernel_size=3)
        ])
        self.first_pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=(0, 5) if train else 0, shear=(0, 5) if train else 0, translate=(0.1, 0.1) if train else None, scale=(0.2, 0.7) if train else None, interpolation=InterpolationMode.BILINEAR, fill=im_mean),
            transforms.GaussianBlur(kernel_size=3)
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=(0, 15) if train else 0, shear=(0, 10) if train else 0, translate=(0.2, 0.2) if train else None, scale=(0.2, 0.7) if train else None, interpolation=InterpolationMode.NEAREST, fill=0),
            transforms.GaussianBlur(kernel_size=3)
        ])
        self.first_pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=(0, 5) if train else 0, shear=(0, 5) if train else 0, translate=(0.1, 0.1) if train else None, scale=(0.2, 0.7) if train else None, interpolation=InterpolationMode.NEAREST, fill=0),
            transforms.GaussianBlur(kernel_size=3)
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(brightness=(0.8,1.2), contrast=(0.8,1.2), saturation=(0.8,1.2), hue=0.15),
            transforms.RandomGrayscale(0.1),
        ])

        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
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
        info = {}
        info['name'] = self.image_list[idx].split('/')[-1]
        info['frames'] = [0]*self.cfg.num_frames
        im = Image.open(self.image_list[idx]).convert('RGB')
        gt = Image.open(self.gt_list[idx]).convert('L')
        sequence_seed = np.random.randint(2147483647)

        cropped_img = []
        images = []
        masks = []

        reseed(sequence_seed)
        this_im = self.all_im_dual_transform(im)
        this_im_ = self.all_im_lone_transform(this_im)  
        
        reseed(sequence_seed)
        this_gt_ = self.all_gt_dual_transform(gt)
        
        for frame_idx in range(self.cfg.num_frames):
            pairwise_seed = np.random.randint(2147483647)
            reseed(pairwise_seed)
            this_im = self.pair_im_dual_transform(this_im_) if frame_idx else self.first_pair_im_dual_transform(this_im_)
            # this_im = self.pair_im_lone_transform(this_im)
            
            this_gt = np.asarray(this_gt_)
            ids = np.unique(this_gt)
            
            each_gt = None
            for each_sep_obj in ids:
                # print(each_sep_obj)
                this = Image.fromarray((this_gt==each_sep_obj).astype(np.uint8))
                reseed(pairwise_seed)
                this = self.pair_gt_dual_transform(this) if frame_idx else self.first_pair_gt_dual_transform(this)
                
                this = np.asarray(this).copy()
                this[this!=0] = each_sep_obj
                
                if each_gt is None:
                    each_gt = this
                each_gt += this
            
            cropped_img.append(pil_to_tensor(this_im))
            this_im = torch.as_tensor(self.resize_longest.apply_image(np.array(this_im, dtype=np.uint8))).permute(2, 0, 1)
            resize_longest_size = this_im.shape[-2:]
            this_im = self.preprocess(this_im)
            images.append(this_im)
            
            if each_gt is None:
                return self.__getitem__(np.random.randint(self.__len__()))
            masks.append(each_gt)

        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)

        target_objects = np.unique(masks[0]).tolist()
        if 0 in target_objects:
            target_objects.remove(0)
        if len(target_objects) > self.cfg.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.cfg.max_num_obj, replace=False)
        
        info['num_objects'] = max(1, len(target_objects))

        # Generate one-hot ground-truth
        H, W = tuple(masks.shape[1:])
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

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("L")

    def __len__(self):
        return len(self.image_list)

def collate_fn(batch):
    output = {}
    for key in batch[0].keys():
        output[key] = [d[key] for d in batch]
        if key in ["image", "prev_masks", "selector"]:
            output[key] = torch.stack(output[key], 0)
    
    return output

# dataloader for training
def get_loader(cfg):
    train_dataset = VideoDataset(cfg, train=True)
    train_data_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            persistent_workers=cfg.persistent_workers,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            collate_fn=collate_fn
    )
    
    return train_data_loader