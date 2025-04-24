import torch
from dataset import PARTNET
from test import SlotAttention
import matplotlib.pyplot as plt
import numpy as np
import random

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = PARTNET(split='train')
img_index = random.randint(0, len(dataset) - 1)
sample = dataset[img_index]
image = sample['image'].unsqueeze(0).to(device)  # Add batch dim

# Load model
resolution = (128, 128)
num_slots = 4
num_iterations = 3
hid_dim = 64
model_dir = './tmp/model1000.ckpt'

model = SlotAttention(
    input_shape=resolution,
    num_slots= num_slots, # opt.num_slots,,
    # slot_size=opt.hid_dim,
    # hidden_dim=opt.hid_dim * 8,
    num_iters=3,     # opt.num_iterations,
    num_channels=3,
).to(device)

checkpoint = torch.load(model_dir,  map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
# Forward pass (no grad)
import time
start = time.time()
with torch.no_grad():
    dict_results = model(image)
end = time.time()
print(f"Time taken for forward pass: {end - start:.4f} seconds")
recon_combined, recons, masks, slots = dict_results['recons_full'], dict_results['recons'], dict_results['masks_dec'], dict_results['slots']
# Prepare visuals
input_img = image.squeeze().cpu().numpy().transpose(1, 2, 0)  # [C,H,W] → [H,W,C]
recon_img = recon_combined.squeeze().detach().cpu().numpy().transpose(1, 2, 0)

# Per-slot reconstructions
slot_imgs = recons.squeeze().detach().cpu().numpy()  # [num_slots, H, W, C]
slot_masks = masks.squeeze().detach().cpu().numpy()  # [num_slots, H, W, 1]
# Compute center of mass for each mask
coms = []
for i in range(num_slots):
    mask = slot_masks[i].squeeze()  # [H, W]
    H, W = mask.shape

    y_coords, x_coords = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
    com_y = (mask * y_coords).sum() / (mask.sum() + 1e-6)
    com_x = (mask * x_coords).sum() / (mask.sum() + 1e-6)

    # Convert from [-1, 1] back to image coordinates
    com_x_img = ((com_x + 1) / 2.0) * W
    com_y_img = ((com_y + 1) / 2.0) * H
    coms.append((com_x_img, com_y_img))

# Plot with COM overlay
fig, axs = plt.subplots(2, num_slots + 2, figsize=(3*(num_slots+2), 6))

# Original and reconstructed image
axs[0, 0].imshow(input_img)
axs[0, 0].set_title("Original Image")
axs[0, 0].axis('off')

axs[0, 1].imshow(recon_img)
axs[0, 1].set_title("Reconstruction")
axs[0, 1].axis('off')

# Per-slot images with COM
for i in range(num_slots):
    masked_img = slot_imgs[i] * slot_masks[i]
    masked_img = np.transpose(masked_img, (1, 2, 0))

    axs[0, i + 2].imshow(masked_img)
    axs[0, i + 2].scatter(coms[i][0], coms[i][1], color='red', s=40, marker='x')  # COM
    axs[0, i + 2].set_title(f"Slot {i+1}")
    axs[0, i + 2].axis('off')

    axs[1, i + 2].imshow(slot_masks[i].squeeze(), cmap='gray')
    axs[1, i + 2].scatter(coms[i][0], coms[i][1], color='red', s=40, marker='x')  # COM
    axs[1, i + 2].set_title(f"Mask {i+1}")
    axs[1, i + 2].axis('off')

# Hide unused subplots
axs[1, 0].axis('off')
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()