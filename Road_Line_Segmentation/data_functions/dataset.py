import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import os
from Road_Line_Segmentation.configs.config import COLORMAP
class RoadLineDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.image_lst = []
        self.mask_lst = []
        for image in os.listdir(os.path.join(root_dir, "images")):
            self.image_lst.append(image)
        for img in self.image_lst:
          if img.endswith(".jpg"):
                mask = img.split(".jpg")[0] + ".png"
          elif img.endswith(".png"):
                mask = img
          self.mask_lst.append(mask)


    def __len__(self):
        return len(self.image_lst)

    def _convert_to_segmentation_mask(self, mask):
        # print(mask)
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(COLORMAP)), dtype=np.float32)
        for label_index, label in enumerate(COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        return segmentation_mask  # (H, W, C)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, "images", self.image_lst[index])
        mask_path = os.path.join(self.root_dir, "masks", self.mask_lst[index])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self._convert_to_segmentation_mask(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        mask = torch.tensor(mask.argmax(axis=2), dtype=torch.long)

        return image, mask
    
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor