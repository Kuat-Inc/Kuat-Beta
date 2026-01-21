# Kuat Private Beta - Quick Start Guide

Welcome to the Kuat private beta! Get 6-25x faster dataset loading in 5 minutes.

## Step 1: Download CLI Encoder

1. Go to [Releases](https://github.com/Kuat-Inc/Kuat-Beta/releases/latest)
2. Download the `quat-tree` binary for your platform:
   - **Linux**: `quat-tree-linux-x64`
   - **macOS Apple Silicon**: `quat-tree-macos-arm64`
   - **macOS Intel**: `quat-tree-macos-x64`
   - **Windows**: `quat-tree-windows-x64.exe`

3. Make it executable (Linux/macOS):
```bash
chmod +x quat-tree-*
```

## Step 2: Encode Your Dataset

Convert your images to a `.kt` archive:

```bash
# ImageNet folders (224x224 images)
./quat-tree vq-create ./imagenet/train -o train.kt --format imagenet -r

# CIFAR-10 (32x32 images) - default size
./quat-tree vq-create ./images -o dataset.kt -r

# Custom size
./quat-tree vq-create ./images -o dataset.kt -r --width 256 --height 256

# Check archive info
./quat-tree vq-info train.kt
```

**Important**: Default is 32x32. For full ImageNet resolution, use `--format imagenet` or specify `--width 224 --height 224`.

## Step 3: Install Python Wheel

1. Go to [Releases](https://github.com/Kuat-Inc/Kuat-Beta/releases/latest)
2. Download the wheel matching your Python version and platform
3. Install:

```bash
pip install kuat-0.1.0-cp311-YOUR_PLATFORM.whl
```

Example platforms:
- Linux x86_64: `manylinux_2_34_x86_64`
- macOS ARM: `macosx_11_0_arm64`
- Windows: `win_amd64`

## Step 4: Train Your Model

Drop-in replacement for PyTorch DataLoader:

```python
from kuat import GPUDataset
import torch
import random

# Load dataset with GPU decode
dataset = GPUDataset("train.kt", device="cuda")

# Training loop
model = YourModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(100):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    for i in range(0, len(indices), 64):
        batch_idx = indices[i:i+64]
        images, labels = dataset[batch_idx]  # Already on GPU!
        
        loss = model(images, labels)
        loss.backward()
        optimizer.step()
```

## Performance Tips

- Use `GPUDataset` for GPU training (fastest)
- Use larger batch sizes when possible
- Archives are O(1) random access - shuffling is free

## Troubleshooting

### "Failed to open archive"
- Verify the `.kt` file path is correct
- Check the file was created successfully with `quat-tree vq-info`

### "module 'kuat' has no attribute..."
- Ensure you installed the wheel matching your Python version
- Try reinstalling: `pip uninstall kuat && pip install kuat-*.whl`

### Encoding takes long
- Large datasets (>100K images) may take several minutes
- Progress is shown - let it complete

### Wrong image dimensions
- Default is 32x32. For ImageNet, use `--format imagenet` or `--width 224 --height 224`

## Feedback

Please open issues at: https://github.com/Kuat-Inc/Kuat-Beta/issues

We'd love to hear:
- What speedup you're seeing vs PyTorch
- Your dataset size and use case
- Any bugs or feature requests

---

Thank you for being a beta tester!
