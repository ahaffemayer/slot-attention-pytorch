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
    def __init__(self, split='train', max_obstacles=3):
        super(PARTNET, self).__init__()
        assert split in ['train', 'val', 'test']
        self.split = split
        self.root_dir = Path.cwd() / "generated_scenes"
        self.img_transform = transforms.Compose([transforms.ToTensor()])
        self.max_obstacles = max_obstacles

        # Load scene metadata
        json_path = self.root_dir / "scenes_data.json"
        with open(json_path, "r") as f:
            self.scenes_data = json.load(f)
        self.scene_keys = sorted(self.scenes_data.keys())

        # Auto-compute min/max bounds for y and z
        self.y_min, self.y_max, self.z_min, self.z_max = self._compute_position_bounds()

    def _compute_position_bounds(self):
        y_values = []
        z_values = []
        for key in self.scene_keys:
            obstacles = self.scenes_data[key].get("obstacles", [])
            for obs in obstacles:
                y, z = obs["position"][1:3]  # Take y and z
                y_values.append(y)
                z_values.append(z)
        
        y_min = min(y_values)
        y_max = max(y_values)
        z_min = min(z_values)
        z_max = max(z_values)

        return y_min, y_max, z_min, z_max

    def normalize_position(self, pos):
        y, z = pos
        y_norm = 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1
        z_norm = 2 * (z - self.z_min) / (self.z_max - self.z_min) - 1
        return [y_norm, z_norm]

    def __getitem__(self, index):
        scene_key = self.scene_keys[index]
        scene_info = self.scenes_data[scene_key]

        img_path = Path(scene_info["image_path"])
        image = Image.open(img_path).convert("RGB")
        image = image.resize((128, 128))
        image = self.img_transform(image)

        raw_obstacles = scene_info.get("obstacles", [])
        positions = [self.normalize_position(obs["position"][1:3]) for obs in raw_obstacles]

        padded_positions = np.zeros((self.max_obstacles, 2), dtype=np.float32)
        for i, pos in enumerate(positions[:self.max_obstacles]):
            padded_positions[i] = pos
        obstacles_tensor = torch.tensor(padded_positions, dtype=torch.float32)

        return {
            'image': image,
            'obstacles': obstacles_tensor,
        }

    def __len__(self):
        return len(self.scene_keys)
