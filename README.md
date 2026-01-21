# Kuat - Ultra-Fast ML Dataset Loader


Kuat loads pre-compressed `.kt` archives with O(1) random access, enabling significantly faster training loops than standard image folders.

> ⚠️ **Private Beta** - This package reads `.kt` archives. Download the `quat-tree` binary to convert your own datasets.

## Installation

```bash
# 1. Install the Python package (download wheel for your platform from Releases)
pip install kuat-0.1.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# 2. Install dependencies
pip install numpy torch  # torch optional, only needed for GPU decode

# 3. Download quat-tree binary for your platform (for encoding datasets)
# See Releases page for:
#   - quat-tree-linux-x64
#   - quat-tree-macos-arm64  (M1/M2)
#   - quat-tree-macos-x64    (Intel)
#   - quat-tree-windows-x64.exe
```

## Converting Your Dataset

Use the `quat-tree` binary to convert image folders to `.kt` archives:

```bash
# ImageNet-style folder (class subfolders)
./quat-tree vq-create ./imagenet/train -o train.kt --width 224 --height 224 -r

# Flat folder of images
./quat-tree vq-create ./my-images -o dataset.kt --width 224 --height 224 -r 

# With custom patch size (default is 2x2)
./quat-tree vq-create ./images -o dataset.kt -r --width 224 --height 224 --patch 4

# Show archive info
./quat-tree vq-info train.kt
```

## Quick Start

### CPU Decode (Simple)

```python
import kuat

# Load archive
archive = kuat.KuatArchive("imagewoof_train.kt")
print(f"Images: {archive.len()}")

# Decode single image
img = archive.decode(0)  # Returns (H, W, C) uint8 numpy array
label = archive.label(0)

# Decode batch
images = archive.decode_batch([0, 1, 2, 3])  # (B, H, W, C)
```

### GPU Decode (Fast - Recommended for Training)

```python
import kuat
from kuat import GPUDecoder, GPUDataset
import torch

# Load archive
archive = kuat.KuatArchive("imagewoof_train.kt")

# Create GPU decoder (uploads codebook once)
decoder = GPUDecoder(archive, device="cuda")

# Decode batch on GPU
indices = torch.tensor(archive.get_indices_batch([0, 1, 2, 3]), device="cuda")
images = decoder.decode(indices)  # (B, C, H, W) float32 on GPU

# Or use the full dataset wrapper
dataset = GPUDataset(archive, device="cuda", normalize=True)
images, labels = dataset[0:64]  # Batch of 64
```

### Training Loop Example

```python
from kuat import GPUDataset
import torch
import random

# Load dataset
dataset = GPUDataset("imagewoof_train.kt", device="cuda", normalize=True)

# Training loop
model = YourModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

indices = list(range(len(dataset)))
for epoch in range(100):
    random.shuffle(indices)
    for i in range(0, len(indices), 64):
        batch_idx = indices[i:i+64]
        images, labels = dataset[batch_idx]
        
        loss = model(images, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## API Reference

### KuatArchive

Low-level archive access:

```python
archive = kuat.KuatArchive("dataset.kt")

# Properties
archive.len()           # Number of images
archive.codebook_size   # Codebook entries (for GPU decode)
archive.patch_size      # Patch dimensions (e.g., 4)
archive.dimensions      # (channels, height, width)
archive.grid_size       # Patches per image (e.g., 3136 for 224x224 with 4x4 patches)

# CPU decode
archive.decode(index)              # Single image (H, W, C)
archive.decode_batch([0, 1, 2])    # Batch (B, H, W, C)
archive.label(index)               # Single label
archive.all_labels()               # All labels as numpy array

# GPU decode support
archive.codebook_numpy()           # Codebook as (N, patch_size) float32
archive.get_indices_batch([0, 1])  # Indices as (B, grid_size) uint32
```

### GPUDecoder

Upload codebook once, decode batches via GPU gather:

```python
decoder = GPUDecoder(
    archive,              # KuatArchive or path string
    device="cuda",        # "cuda" or "cpu"
    dtype=torch.float32,  # Output dtype
)

# Decode indices to images
images = decoder.decode(indices)  # (B, C, H, W)
```

### GPUDataset

Full dataset with labels, ready for training:

```python
dataset = GPUDataset(
    archive,              # KuatArchive or path string
    device="cuda",
    normalize=True,       # ImageNet normalization
    layout="NCHW",        # or "NHWC"
)

images, labels = dataset[0:64]      # Slice indexing
images, labels = dataset[[0, 5, 10]]  # List indexing
```

## Performance

Benchmarked on ImageWoof (9,025 images, 224×224):

| Method | Images/sec | vs PyTorch |
|--------|------------|------------|
| PyTorch ImageFolder | 691 | 1.0x |
| Kuat CPU (batch=64) | 4,168 | **6.0x** |
| Kuat CPU (batch=512) | 6,252 | **9.0x** |

## Requirements

- Python 3.9-3.12
- NumPy
- PyTorch (optional, for GPU decode)

## Support

For beta access or questions:
- GitHub Issues on this repo
- founders@kuatlabs.com
## License

Proprietary - Kuat Inc. All rights reserved.
