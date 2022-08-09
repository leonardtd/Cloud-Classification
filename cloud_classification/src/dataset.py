import os

import torch
import torchvision.transforms as T
from torchvision.io import read_image

from . import utils

### Extracted from EDA
MEAN_R = 130.9360
MEAN_G = 160.3594
MEAN_B = 193.0078

STD_R = 23.4079
STD_G = 20.6422
STD_B = 18.6495


class GCD:
    def __init__(self, image_paths, resize=None, use_augmentation=False):

        self.image_paths  = image_paths
        self.targets = utils.get_gcd_targets(image_paths)
        self.resize = resize
        self.use_augmentation = use_augmentation
        
        self.meanR = MEAN_R
        self.meanG = MEAN_G
        self.meanB = MEAN_B
        
        self.stdR = STD_R
        self.stdG = STD_G
        self.stdB = STD_B
        
        ### Preprocessing
        self.norm_transform = T.Normalize(mean=[self.meanR, self.meanG, self.meanB], std=[self.stdR, self.stdG, self.stdB])
        
        ### Augmentation
        self.aug_transform = T.Compose([
                                      T.RandomHorizontalFlip(p=0.5),
                                      T.RandomVerticalFlip(p=0.5),
                                      T.RandomRotation((-12,12)),
                              ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = read_image(self.image_paths[item]).float()
        
        #Normalize by channel
        image = self.norm_transform(image)      
        target = torch.tensor(self.targets[item], dtype=torch.long)

        if self.resize is not None:
            image = T.Resize(self.resize)(image)
            
        if self.use_augmentation:
            image = self.aug_transform(image)

        return {
            "images": image,
            "targets": target,
        }
    
    
    
    