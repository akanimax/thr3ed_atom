import math
from typing import Any

import torch
from torch import Tensor

from thre3d_atom.utils.constants import INFINITY


def mse2psnr(x: Any) -> Any:
    if isinstance(x, Tensor):
        dtype, device = x.dtype, x.device
        # fmt: off
        return (
            -10.0 * torch.log(x) / torch.log(torch.tensor([10.0], dtype=dtype, device=device))
            if x != 0.0
            else torch.tensor([INFINITY], dtype=dtype, device=device)
        )
        # fmt: on
    else:
        return -10.0 * math.log(x) / math.log(10.0) if x != 0.0 else math.inf
