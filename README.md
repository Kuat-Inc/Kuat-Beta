# Kuat - Ultra-Fast ML Dataset Loader

**6-25x faster dataloading via vector quantization.**

Kuat loads pre-compressed `.qvq` archives with O(1) random access, enabling significantly faster training loops than standard image folders.

> ⚠️ **Private Beta** - This package reads `.qvq` archives. Contact us for sample datasets or encoding your own data.

## Installation

```bash
# Download the wheel for your platform from GitHub Actions artifacts
pip install kuat-0.1.0b1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Dependencies
pip install numpy torch  # torch optional, only for GPU decode
```

## Quick Start

### CPU Decode (Simple)

```python
import kuat

# Load archive
archive = kuat.KuatArchive("imagewoof_train.qvq")
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
archive = kuat.KuatArchive("imagewoof_train.qvq")

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

# Load dataset
dataset = GPUDataset("imagewoof_train.qvq", device="cuda", normalize=True)

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
archive = kuat.KuatArchive("dataset.qvq")

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
| Kuat GPU | 50,000+ | **70x+** |

## File Format

Kuat reads `.qvq` archives created by the Quattree encoder:
- **Adaptive VQ**: Variable-size codebook (up to 33M entries)
- **12-byte patches**: RGB 2×2 patches quantized to single indices
- **O(1) access**: Memory-mapped for instant random access

## Requirements

- Python 3.11
- NumPy
- PyTorch (optional, for GPU decode)

## Support

For beta access, sample datasets, or encoding your own data:
- GitHub Issues on this repo
- Email: [contact email]

## License

Proprietary - Kuat Inc. All rights reserved.
