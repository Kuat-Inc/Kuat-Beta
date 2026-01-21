# Kuat - Ultra-Fast ML Dataset Compression

**10x faster dataloading through intelligent compression.**

Kuat compresses your ML datasets (ImageNet, CIFAR, custom) into a compact format with O(1) random access, enabling significantly faster training loops.

## Quick Start

### Installation

```bash
# Download the wheel for your platform from the release
pip install kuat-0.1.0b1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

### Convert Your Dataset

```bash
# ImageNet-style folder structure
kuat convert ./imagenet/train train.kuat --format imagenet

# CIFAR-10
kuat convert ./cifar-10-batches-py cifar10.kuat --format cifar

# Generic image folder
kuat convert ./my-images dataset.kuat --format images
```

### Use in Training

```python
from kuat import KuatDataset

# Create dataset
dataset = KuatDataset("train.kuat", batch_size=64, shuffle=True)

# Training loop
for epoch in range(100):
    for batch in dataset.epoch(epoch):
        images = batch["images"]  # (B, H, W, C) uint8 numpy array
        labels = batch["labels"]  # (B,) int32 numpy array
        
        # Convert to PyTorch tensors
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        labels = torch.from_numpy(labels).long()
        
        # Your training code here
        ...
```

### With PyTorch DataLoader (optional)

```python
import torch
from kuat import KuatDataset

class KuatPyTorchDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, batch_size=64, shuffle=True):
        self.dataset = KuatDataset(path, batch_size=batch_size, shuffle=shuffle)
        self.epoch = 0
    
    def __iter__(self):
        for batch in self.dataset.epoch(self.epoch):
            images = torch.from_numpy(batch["images"]).permute(0, 3, 1, 2).float() / 255.0
            labels = torch.from_numpy(batch["labels"]).long()
            yield images, labels
        self.epoch += 1

# Use with standard DataLoader
dataset = KuatPyTorchDataset("train.kuat", batch_size=64)
loader = torch.utils.data.DataLoader(dataset, batch_size=None)

for images, labels in loader:
    ...
```

## CLI Reference

```bash
# Convert dataset
kuat convert <input> <output> [options]
  --format, -f    Input format: auto, imagenet, cifar, images
  --bits, -b      Quantization bits 1-8 (default: 4)
  --codebook-size Max codebook entries (default: 512)
  --width, -W     Target image width
  --height, -H    Target image height
  --quiet, -q     Suppress progress output

# Show archive info
kuat info <archive>

# Benchmark dataloader speed
kuat benchmark <archive> [--batch-size 64] [--epochs 1]
```

## Python API

### KuatArchive

Low-level archive access:

```python
from kuat import KuatArchive

archive = KuatArchive("dataset.kuat")
print(len(archive))  # Number of samples
print(archive.info())  # Archive metadata
```

### KuatDataset

High-level PyTorch-compatible dataset:

```python
from kuat import KuatDataset

dataset = KuatDataset(
    "train.kuat",
    batch_size=64,      # Batch size
    shuffle=True,       # Shuffle each epoch
    seed=42,            # Random seed
    drop_last=False,    # Drop incomplete last batch
    layout="NHWC",      # Memory layout: NHWC or NCHW
)

# Iterate by epoch
for batch in dataset.epoch(0):
    images = batch["images"]  # (B, H, W, C) or (B, C, H, W)
    labels = batch["labels"]  # (B,)
```

### convert_dataset

Programmatic conversion:

```python
from kuat import convert_dataset

stats = convert_dataset(
    "imagenet/train",
    "train.kuat",
    format="imagenet",
    bits=4,
    codebook_size=512,
    width=224,
    height=224,
)
print(f"Converted {stats['samples']} samples")
print(f"Compression: {stats['compression_ratio']:.1f}x")
```

## Supported Formats

| Format | Description | Auto-detect |
|--------|-------------|-------------|
| `imagenet` | ImageNet-style folders (class/image.jpg) | ✓ |
| `cifar` | CIFAR-10/100 pickle batches | ✓ |
| `images` | Flat folder of images | ✓ |

## Performance

Typical benchmarks on ImageNet:

| Metric | PyTorch ImageFolder | Kuat |
|--------|---------------------|------|
| Load time per batch | 50-100ms | 5-10ms |
| Disk I/O | 150 MB/s | 50 MB/s |
| Memory pressure | High | Low |
| Random access | Slow (seek) | O(1) |

## Support

For issues or questions during the beta, contact: [your email]

## License

Proprietary - Kuat Inc. All rights reserved.
