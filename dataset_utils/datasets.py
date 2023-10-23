import os
import glob
#import random
from typing import List, Union, Dict, Any

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop

class LSDMDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, os.PathLike],
        target_subdirs: List[Union[str, os.PathLike]],
        caption_subdir: Union[str, os.PathLike],
        condition_subdir: Union[str, os.PathLike],
        resolution: int = 512,
        proportion_empty_prompts: float = 0.0,
        #center_crop: bool = True,
        #random_flip: bool = False,
        ext: Union[str, List[str]] = "png",
        seed: int = 1,
    ):
        self.cond_dir = os.path.join(data_dir, condition_subdir)
        self.tgt_dirs = [os.path.join(data_dir, sub_dir) for sub_dir in target_subdirs]
        self.cap_dir = os.path.join(data_dir, caption_subdir)
        self.resolution = resolution

        # get image filenames
        search_dir = os.path.join(data_dir, target_subdirs[0])
        exts = [ext] if isinstance(ext, str) else ext
        self.img_names = []
        for ext in exts:
            img_names = [
                os.path.basename(f) for f in glob.glob(os.path.join(search_dir, f"*.{ext}"))
            ]
            self.img_names.extend(img_names)
        self.img_names = sorted(self.img_names)

        self.random_crop = transforms.RandomCrop(resolution)
        self.preproc = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                #transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        
        self.rng = np.random.default_rng(seed)
        self.proportion_empty_prompts = proportion_empty_prompts
    
    def __len__(self) -> int:
        return len(self.img_names)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = dict()

        # get data paths
        img_name = self.img_names[idx]
        tgt_paths = [os.path.join(dir, img_name) for dir in self.tgt_dirs]
        cond_path = os.path.join(self.cond_dir, img_name)
        cap_name = ".".join([*(img_name.split(".")[:-1]), "txt"])
        cap_path = os.path.join(self.cap_dir, cap_name)

        # load images
        tgt_imgs = [Image.open(path).convert("RGB") for path in tgt_paths]
        cond_img = Image.open(cond_path).convert("RGB")

        # get image size and crop coordinates
        ref_img = tgt_imgs[0]
        original_size = (ref_img.width, ref_img.height)
        top, left, h, w = self.random_crop.get_params(ref_img, (self.resolution, self.resolution))
        crop_top_left = (top, left)
        # crop images
        tgt_imgs = [crop(img, top, left, h, w) for img in tgt_imgs]
        cond_img = crop(cond_img, top, left, h, w)

        example["pixel_values"] = torch.stack([self.preproc(img) for img in tgt_imgs])
        example["conditioning_pixel_values"] = self.preproc(cond_img)
        example["original_size"] = original_size
        example["crop_top_left"] = crop_top_left

        # get prompt
        with open(cap_path, "r", encoding="utf-8") as f:
            prompt = f.readline()
        # randomly replace the caption with empty string
        example["prompt"] = (
            "" if self.rng.random() < self.proportion_empty_prompts else prompt
        )

        return example

class SGRDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, os.PathLike],
        target_subdir: Union[str, os.PathLike],
        condition_subdirs: List[Union[str, os.PathLike]],
        caption_subdir: Union[str, os.PathLike],
        resolution: int = 512,
        #proportion_robust_conditioning: float = 0.5,
        #center_crop: bool = True,
        #random_flip: bool = False,
        ext: Union[str, List[str]] = "png",
        seed: int = 1,
    ):
        self.tgt_dir = os.path.join(data_dir, target_subdir)
        self.cond_dirs = [os.path.join(data_dir, sub_dir) for sub_dir in condition_subdirs]
        self.cap_dir = os.path.join(data_dir, caption_subdir)
        self.resolution = resolution

        # get image filenames
        search_dir = os.path.join(data_dir, target_subdir)
        exts = [ext] if isinstance(ext, str) else ext
        self.img_names = []
        for ext in exts:
            img_names = [
                os.path.basename(f) for f in glob.glob(os.path.join(search_dir, f"*.{ext}"))
            ]
            self.img_names.extend(img_names)
        self.img_names = sorted(self.img_names)

        self.random_crop = transforms.RandomCrop(resolution)
        self.preproc = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                #transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.rng = np.random.default_rng(seed)
        #self.proportion_robust_conditioning = proportion_robust_conditioning
        self.num_conds = 1 + len(condition_subdirs)
    
    def __len__(self) -> int:
        return len(self.img_names)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = dict()

        # get data paths
        img_name = self.img_names[idx]
        tgt_path = os.path.join(self.tgt_dir, img_name)
        cond_paths = [os.path.join(dir, img_name) for dir in self.cond_dirs]
        cap_name = ".".join([*(img_name.split(".")[:-1]), "txt"])
        cap_path = os.path.join(self.cap_dir, cap_name)

        # load images
        tgt_img = Image.open(tgt_path).convert("RGB")
        cond_imgs = [Image.open(path).convert("RGB") for path in cond_paths]

        # get image size and crop coordinates
        ref_img = tgt_img
        original_size = (ref_img.width, ref_img.height)
        top, left, h, w = self.random_crop.get_params(ref_img, (self.resolution, self.resolution))
        crop_top_left = (top, left)
        # crop images
        tgt_img = crop(tgt_img, top, left, h, w)
        cond_imgs = [crop(img, top, left, h, w) for img in cond_imgs]

        tgt_img = self.preproc(tgt_img)
        cond_imgs = [self.preproc(img) for img in cond_imgs]

        # get prompt
        with open(cap_path, "r", encoding="utf-8") as f:
            prompt = f.readline()

        # robust conditioning, randomly dropput conditions
        # In the parer (https://arxiv.org/abs/2310.08579), it is described as
        # "we randomly mask out any of the control signals, such as replace text prompt with empty string,
        #  or substitute the structural maps with zero-value images."
        # but, I don't know the exact process to choose condition(s) to dropout

        # choose the number of conditions to dropout
        num_dropouts = self.rng.integers(0, high=self.num_conds + 1)
        # get indices of dropout conditions, `self.num_conds - 1` is prompt, images otherwise
        dropout_conds = self.rng.choice(
            [i for i in range(self.num_conds)], size=num_dropouts, replace=False
        )
        # dropout conditions
        for cond in dropout_conds:
            if cond == self.num_conds - 1:
                # for prompts, make it empty string
                prompt = ""
            else:
                # for images, fill values with zero
                cond_imgs[cond] = torch.zeros_like(cond_imgs[cond], dtype=cond_imgs[cond].dtype)
        
        example["pixel_values"] = tgt_img
        example["conditioning_pixel_values"] = torch.stack(cond_imgs)
        example["original_size"] = original_size
        example["crop_top_left"] = crop_top_left
        example["prompt"] = prompt

        return example
