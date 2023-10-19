import os
import glob
import random
from typing import List, Union, Dict, Any

from PIL import Image

#import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LSDMDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, os.PathLike],
        condition_subdir: Union[str, os.PathLike],
        caption_subdir: Union[str, os.PathLike],
        target_subdirs: List[Union[str, os.PathLike]],
        resolution: int = 512,
        center_crop: bool = True,
        proportion_empty_prompts: float = 0.0,
        #random_flip: bool = False,
        ext: Union[str, List[str]] = "png"
    ):
        self.cond_dir = os.path.join(data_dir, condition_subdir)
        self.cap_dir = os.path.join(data_dir, caption_subdir)
        self.tgt_dirs = [os.path.join(data_dir, sub_dir) for sub_dir in target_subdirs]

        # get image filenames
        search_dir = os.path.join(data_dir, target_subdirs[0])
        exts = [ext] if isinstance(ext, str) else ext
        self.img_names = []
        for ext in exts:
            img_names = [
                os.path.basename(f) for f in glob.glob(os.path.join(search_dir, f"*.{ext}"))
            ]
            self.img_names.extend(sorted(img_names))
        self.img_names = sorted(img_names)

        self.preproc = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
                #transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.proportion_empty_prompts = proportion_empty_prompts
    
    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = dict()

        # get data paths
        img_name = self.img_names[idx]
        cond_path = os.path.join(self.cond_dir, img_name)
        tgt_paths = [os.path.join(dir, img_name) for dir in self.tgt_dirs]
        cap_name = ".".join([*(img_name.split(".")[:-1]), "txt"])
        cap_path = os.path.join(self.cap_dir, cap_name)

        # load images
        cond_img = Image.open(cond_path).convert("RGB")
        tgt_imgs = [Image.open(path).convert("RGB") for path in tgt_paths]
        example["pixel_values"] = self.preproc(cond_img)
        example["tgt_pixel_values"] = [self.preproc(img) for img in tgt_imgs]

        # get caption
        with open(cap_path, "r", encoding="utf-8") as f:
            caption = f.readline()[0]
        example["prompt"] = (
            "" if random.random() < self.proportion_empty_prompts else caption   
        )

        return example
