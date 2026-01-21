# Kuat Private Beta - Quick Start Guide

Welcome to the Kuat private beta! This guide will get you up and running in 5 minutes.

## 1. Install the Package

Download the wheel file for your platform from the release:

| Platform | Python | File |
|----------|--------|------|
| Linux x86_64 | 3.11 | `kuat-0.1.0b1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` |
| Linux x86_64 | 3.12 | `kuat-0.1.0b1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` |
| macOS Intel | 3.11 | `kuat-0.1.0b1-cp311-cp311-macosx_10_12_x86_64.whl` |
| macOS Apple Silicon | 3.11 | `kuat-0.1.0b1-cp311-cp311-macosx_11_0_arm64.whl` |
| Windows x64 | 3.11 | `kuat-0.1.0b1-cp311-none-win_amd64.whl` |

```bash
# Create a virtual environment (recommended)
python -m venv kuat-env
source kuat-env/bin/activate  # Linux/macOS
# or: kuat-env\Scripts\activate  # Windows

# Install the wheel
pip install /path/to/kuat-0.1.0b1-*.whl

# Verify installation
kuat --version
```

## 2. Convert Your Dataset

### ImageNet-style folders

If your dataset is organized as `train/class_name/image.jpg`:

```bash
kuat convert ./my-dataset/train train.kuat --format imagenet
kuat convert ./my-dataset/val val.kuat --format imagenet
```

### CIFAR-10/100

```bash
kuat convert ./cifar-10-batches-py cifar10.kuat --format cifar
```

### Flat folder of images

```bash
kuat convert ./images dataset.kuat --format images
```

### Custom dimensions

```bash
# For 32x32 images
kuat convert ./data small.kuat --width 32 --height 32

# For 224x224 images (default for imagenet)
kuat convert ./data large.kuat --width 224 --height 224
```

## 3. Use in Your Training Script

Replace your PyTorch ImageFolder/DataLoader with:

```python
from kuat import KuatDataset

# Create dataset
train_data = KuatDataset("train.kuat", batch_size=64, shuffle=True)

# Training loop - epochs are explicit
for epoch in range(100):
    for batch in train_data.epoch(epoch):
        # images: numpy array (B, H, W, C) uint8
        # labels: numpy array (B,) int32
        images = batch["images"]
        labels = batch["labels"]
        
        # Convert to PyTorch if needed
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        labels = torch.from_numpy(labels).long()
        
        # Your training code...
```

## 4. Benchmark Your Speedup

```bash
# Check how fast batches load
kuat benchmark train.kuat --batch-size 64 --epochs 3
```

Compare with your current DataLoader!

## Troubleshooting

### "Failed to open archive"
- Check the file path is correct
- Verify the .kuat file was created successfully

### "module 'kuat' has no attribute..."
- Make sure you installed the correct wheel for your Python version
- Try `pip uninstall kuat && pip install ...` to reinstall

### Slow conversion
- Large datasets (>100K images) may take several minutes
- Use `--quiet` to reduce console output overhead

## Feedback

We'd love to hear about:
- Your use case and dataset size
- Speed improvements you're seeing
- Any issues or crashes
- Feature requests

Contact: [your-email@example.com]

---

Thank you for being a beta tester! 🚀
