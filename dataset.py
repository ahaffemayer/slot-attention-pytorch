import os
import random
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class PARTNET(Dataset):
    def __init__(self, split='train'):
        super(PARTNET, self).__init__()
        
        assert split in ['train', 'val', 'test']
        self.split = split
        self.root_dir = Path(Path.cwd()) / "generated_scenes/"
        self.img_transform = transforms.Compose([
               transforms.ToTensor()])
        
                # Only include .png files
        self.files = [f for f in os.listdir(self.root_dir) if f.endswith(".png")]
        self.files.sort()  # optional: for consistent ordering

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.files[index])
        image = Image.open(img_path).convert("RGB")
        image = image.resize((128 , 128))
        image = self.img_transform(image)
        sample = {'image': image}

        return sample
            
    
    def __len__(self):
        return len(self.files)