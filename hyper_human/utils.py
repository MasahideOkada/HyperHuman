from typing import Tuple, List

import torch
from torch import Tensor

def get_mean_and_std(x: Tensor, dim: Tuple[int] = (2, 3)) -> Tuple[Tensor]:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    return mean, std

def normalize(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return (x - mean) / std

def average(inputs: List[Tensor]) -> Tensor:
    avg = torch.stack(inputs).sum(dim=0)
    return avg / len(inputs)
