# Kuat - Ultra-Fast ML Dataset Loader

Kuat compresses ML datasets while making them decode 6-25x faster than PyTorch. Training loops run faster with lower storage costs.

## Quick Start

### Step 1: Download CLI Binary

Download the `quat-tree` encoder for your platform from [Releases](https://github.com/Kuat-Inc/Kuat-Beta/releases):

- **Linux**: `quat-tree-linux-x64`
- **macOS Apple Silicon (M1/M2)**: `quat-tree-macos-arm64`
- **macOS Intel**: `quat-tree-macos-x64`
- **Windows**: `quat-tree-windows-x64.exe`

Make it executable (Linux/macOS):
```bash
chmod +x quat-tree-*
```

### Step 2: Encode Your Dataset

Convert your images to a `.kt` archive:

```bash
# ImageNet-style folders (224x224 images)
./quat-tree vq-create ./imagenet/train -o train.kt --format imagenet -r

# CIFAR-10 (32x32 images)
./quat-tree vq-create ./images -o dataset.kt -r --width 32 --height 32

# Custom size
./quat-tree vq-create ./images -o dataset.kt -r --width 224 --height 224

# Check archive info
./quat-tree vq-info train.kt
```

**Note**: Default is 32x32. For ImageNet at full resolution, use `--format imagenet` or specify `--width 224 --height 224`.

### Step 3: Install Python Wheel

Download the wheel for your platform from [Releases](https://github.com/Kuat-Inc/Kuat-Beta/releases):

```bash
pip install kuat-0.1.0-cp311-YOUR_PLATFORM.whl
```

Platform examples:
- Linux: `manylinux_2_34_x86_64` or `manylinux_2_34_aarch64`
- macOS: `macosx_11_0_arm64` or `macosx_10_12_x86_64`
- Windows: `win_amd64`

### Step 4: Train Your Model

Use the archive in your training code:

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
