import numpy as np
import torch
from torchvision import transforms

imagenet_denormalize = transforms.Compose([
    transforms.Normalize(mean = [0., 0., 0.], std = [1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean = [-0.485, -0.456, -0.406], std = [1., 1., 1.])
    ])

def set_seeds(sid=115):
    np.random.seed(sid)

    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return transforms.functional.pad(image, padding, padding_mode = 'edge')
