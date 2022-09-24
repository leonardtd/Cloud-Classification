import os

import torch
import torchvision.transforms as T
from torchvision.io import read_image
from tqdm import tqdm

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
        # self.aug_transform = T.Compose([
        #                               T.RandomHorizontalFlip(p=0.5),
        #                               T.RandomVerticalFlip(p=0.5),
        #                               T.RandomRotation((-12,12)),
        #                       ])
        
        self.aug_transform = T.Compose([
                                T.TrivialAugmentWide(), 
                                #T.RandomErasing(0.1),
                            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = read_image(self.image_paths[item])
        
        if self.resize is not None:
            image = T.Resize(self.resize)(image)

        if self.use_augmentation:
            image = self.aug_transform(image)
        
        #Normalize by channel
        image = self.norm_transform(image.float())
        
        target = torch.tensor(self.targets[item], dtype=torch.long)

        return {
            "images": image,
            "targets": target,
        }
    
    
    
class GCDv2:
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
        
        
        ### reads images
        self._get_data()
        
    def _get_data(self):
        images = []
        final_targets = []
        
        print("Loading data to memory")
        for path, target in tqdm(zip(self.image_paths, self.targets), total=len(self.image_paths)):
            image = read_image(path).float()
            
            #Normalize by channel
            image = self.norm_transform(image)
            
            if self.resize is not None:
                image = T.Resize(self.resize)(image)
                
            images.append(image)
            final_targets.append(target)
            
            if self.use_augmentation:
                image = self.aug_transform(image)
                
                images.append(image)
                final_targets.append(target)
                
        self.images = torch.stack(images, dim=0)
        self.final_targets = torch.tensor(final_targets, dtype=torch.long)
        

    def __len__(self):
        return len(self.final_targets)

    def __getitem__(self, item):
        image = self.images[item]
        target = self.final_targets[item]

        return {
            "images": image,
            "targets": target,
        }
    
    
    