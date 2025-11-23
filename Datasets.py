
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from json_parse import get_dataset
import cv2
import numpy as np

# Define joint transforms
transform = A.Compose([
     A.Resize(3000,3000),
    A.Rotate(45,p=0.5),
    A.RandomCrop(width=2000, height=1500, p=1),
    A.HorizontalFlip(p=0.6),
    A.VerticalFlip(p=0.6),
   
    ToTensorV2()
])


def random_square_mask(image, min_size=200, max_size=None, num_squares=1, p=1.0, fill=0, return_mask=False, inplace=False):
    """Apply one or more random square masks to an image.

    Supports numpy arrays, PIL Images and torch tensors. Returns the masked image
    and optionally the binary mask (1 where masked).

    Parameters
    - image: numpy array HxW or HxWxC, PIL.Image, or torch tensor CxHxW / HxW / BxCxHxW
    - min_size, max_size: range of square side lengths in pixels
    - num_squares: how many random squares to draw
    - p: probability to apply masking
    - fill: fill value for masked pixels
    - return_mask: if True, return (masked_image, mask)
    - inplace: if True attempt to modify input (only for numpy)
    """
    import numpy as _np
    import random

    if random.random() > p:
        if return_mask:
            if isinstance(image, torch.Tensor):
                # create zero mask with spatial dims
                if image.dim() >= 3 and image.shape[0] <= 4:
                    h, w = image.shape[1], image.shape[2]
                else:
                    h, w = image.shape[-2], image.shape[-1]
                return image, torch.zeros((h, w), dtype=torch.uint8)
            else:
                h, w = image.shape[0], image.shape[1]
                return image, _np.zeros((h, w), dtype=_np.uint8)
        return image

    def _mask_numpy(arr):
        h, w = arr.shape[0], arr.shape[1]
        mx = _np.zeros((h, w), dtype=_np.uint8)
        max_s = max_size if max_size is not None else min(h, w) // 2
        max_s = min(max_s, min(h, w))
        for _ in range(num_squares):
            s = _np.random.randint(min_size, max(min_size+1, max_s+1))
            top = _np.random.randint(0, max(1, h - s + 1))
            left = _np.random.randint(0, max(1, w - s + 1))
            mx[top:top+s, left:left+s] = 1
            if arr.ndim == 3:
                arr[top:top+s, left:left+s, :] = fill
            else:
                arr[top:top+s, left:left+s] = fill
        return arr, mx

    def _mask_torch(t):
        # operate on clone to avoid modifying original unless requested
        if not inplace:
            t = t.clone()
        # support BxCxHxW, CxHxW, HxW
        if t.dim() == 4:
            b, c, h, w = t.shape
            target = t[0]
        elif t.dim() == 3 and t.shape[0] <= 4:
            c, h, w = t.shape
            target = t
        elif t.dim() == 2:
            h, w = t.shape
            c = None
            target = t
        else:
            # handle HxWxC torch (uncommon)
            if t.dim() == 3:
                h, w, c = t.shape
                target = t.permute(2, 0, 1)
            else:
                raise ValueError(f'Unsupported tensor shape: {t.shape}')

        max_s = max_size if max_size is not None else min(h, w) // 2
        max_s = min(max_s, min(h, w))
        mask_np = _np.zeros((h, w), dtype=_np.uint8)
        for _ in range(num_squares):
            s = _np.random.randint(min_size, max(min_size+1, max_s+1))
            top = _np.random.randint(0, max(1, h - s + 1))
            left = _np.random.randint(0, max(1, w - s + 1))
            mask_np[top:top+s, left:left+s] = 1

        mask_t = torch.from_numpy(mask_np).to(torch.uint8).to(target.device)
        # apply fill
        if c is None:
            target[mask_np == 1] = fill
        else:
            # channel first
            for ch in range(target.shape[0]):
                target[ch][mask_np == 1] = fill

        # put back into original tensor shape
        if t.dim() == 4:
            t[0] = target
        elif t.dim() == 3 and t.shape[0] <= 4:
            t = target
        elif t.dim() == 2:
            t = target

        return t, mask_t

    # dispatch
    if isinstance(image, torch.Tensor):
        masked, mask = _mask_torch(image)
        if return_mask:
            return masked, mask
        return masked
    else:
        # PIL -> numpy
        if hasattr(image, 'convert'):
            arr = _np.array(image)
        else:
            arr = image
        masked_arr, mask = _mask_numpy(arr if inplace else arr.copy())
        if return_mask:
            return masked_arr, mask
        return masked_arr




def extract_breast_roi(img_array, threshold_value=10):
    """
    Extracts the breast region of interest (ROI) from a mammogram image.
    
    Parameters:
        img_array (numpy.ndarray): Input grayscale mammogram image.
        threshold_value (int): Threshold for separating tissue from background.
        
    Returns:
        roi_cropped (numpy.ndarray): Cropped ROI image with background set to black.
    """
    # Step 1: Normalize to [0,255]
    norm = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)

    # Step 2: Threshold to binary mask
    _, thresh = cv2.threshold(norm, threshold_value, 255, cv2.THRESH_BINARY)

    # Step 3: Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    # Step 4: Find largest component (excluding background)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.uint8(labels == largest_label) * 255

    # Step 5: Apply mask to original normalized image
    roi = cv2.bitwise_and(norm, norm, mask=mask)

    # Step 6: Crop bounding box
    x, y, w, h, area = stats[largest_label]
    roi_cropped = roi[y:y+h, x:x+w]

    return roi_cropped

class ImageMaskDataset(Dataset):
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary containing image and mask paths.
            transform (callable, optional): Transform to apply to images.
            mask_transform (callable, optional): Transform to apply to masks.
        """
        self.data_pairs = []
     

        # Group image-mask pairs by UUID + view + side
        for key in data_dict.keys():
            if key.endswith("_image"):
                base_key = key.replace("_image", "")
                mask_key = base_key + "_mask"
                if mask_key in data_dict:
                    self.data_pairs.append((data_dict[key], data_dict[mask_key]))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image, mask = self.data_pairs[idx]

        
        augmented = transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']
    






class MaskedDataset(Dataset):
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary containing image and mask paths.
            transform (callable, optional): Transform to apply to images.
            mask_transform (callable, optional): Transform to apply to masks.
        """
        self.data_pairs = []
     

        # Group image-mask pairs by UUID + view + side
        for key in data_dict.keys():
            if key.endswith("_image"):
                base_key = key.replace("_image", "")
                mask_key = base_key + "_mask"
                if mask_key in data_dict:
                    self.data_pairs.append((data_dict[key], data_dict[mask_key]))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image, mask = self.data_pairs[idx]
        image = extract_breast_roi(image, threshold_value=10)
        masked_image = random_square_mask(image, min_size=200, max_size=500, num_squares=2)
        

        augmented = transform(image=image, mask=masked_image)
        return augmented['mask'], augmented['image']
    




















if __name__ == "__main__":
        img_dir = "/c:/Users/alqud/Desktop/2025/Retina_classification/Retina_seg/training"  # update path if needed
        dataset = MaskedDataset(get_dataset())
        
        trainloader = DataLoader(dataset, batch_size=1, shuffle=False)

        for i, (image, mask) in enumerate(trainloader):
            if i >= 3:
                break
            img_np = image.squeeze().numpy()
            mask_np = mask.squeeze().numpy()
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.title("Masked Image")
            plt.imshow(img_np, cmap='gray')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.title("Image")
            plt.imshow(mask_np,cmap='gray')
            plt.axis('off')
            plt.show()