"""Utils for pytorch environment.
"""

import torch
import torch.cuda
import torch.backends.cudnn
import numpy as np
import random


def set_random_seed(seed: int):
    """Set random seed for reproducing.

    Args:
        seed: random seed value.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
