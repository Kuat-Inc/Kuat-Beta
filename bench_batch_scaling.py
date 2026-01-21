#!/usr/bin/env python3
"""Analyze Kuat performance bottleneck"""

import time
import os

# Check rayon thread count
print("=" * 60)
print("System Info")
print("=" * 60)
print(f"CPU cores: {os.cpu_count()}")

from kuat import KuatArchive, KuatDataset

qvq_path = r"C:\Users\yaweh\OneDrive\Documents\Kuat Inc Code\Quattree\imagewoof_4bit_adaptive_33m.qvq"

# Test different batch sizes to find optimal
print("\n" + "=" * 60)
print("Batch Size Scaling Test")
print("=" * 60)

for batch_size in [16, 32, 64, 128, 256, 512]:
    dataset = KuatDataset(qvq_path, batch_size=batch_size, shuffle=True)
    
    # Warmup
    for i, batch in enumerate(dataset.epoch(0)):
        if i >= 3:
            break
    
    # Timed run - measure time for ~6400 images
    target_images = 6400
    num_batches = target_images // batch_size
    
    start = time.perf_counter()
    total_images = 0
    for i, batch in enumerate(dataset.epoch(1)):
        total_images += batch['images'].shape[0]
        if i >= num_batches:
            break
    elapsed = time.perf_counter() - start
    
    ips = total_images / elapsed
    print(f"Batch {batch_size:3d}: {total_images:,} images in {elapsed:.2f}s = {ips:,.0f} images/sec")

print("\n" + "=" * 60)
print("Analysis")
print("=" * 60)
print("""
Kuat already uses rayon (Rust's parallel library) internally:
- Each batch decodes images in parallel across all CPU cores
- The decode step is a simple table lookup (very fast)
- Memory bandwidth is likely the bottleneck, not CPU

Unlike PyTorch, Kuat doesn't need "workers" because:
1. No file I/O to parallelize (single mmap file)
2. No Python GIL to work around
3. Decode is already fully parallel in Rust

To go faster, you'd need:
- GPU-accelerated decode (CUDA codebook gather)
- Or load directly to GPU memory
""")
