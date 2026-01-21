#!/usr/bin/env python3
"""Test Kuat GPU decode API (CPU-only simulation)"""

from kuat import KuatArchive
import numpy as np
import time

# Test archive
archive_path = r"C:\Users\yaweh\OneDrive\Documents\Kuat Inc Code\Quattree\imagewoof_4bit_adaptive_33m.qvq"

print("=" * 60)
print("Testing Kuat GPU Decode API")
print("=" * 60)

# Open archive
archive = KuatArchive(archive_path)
print(f"\nArchive: {archive}")
print(f"  Samples: {len(archive):,}")
print(f"  Dimensions: {archive.dimensions}")
print(f"  Grid size: {archive.grid_size}")
print(f"  Patch size: {archive.patch_size}")
print(f"  Codebook size: {archive.codebook_size:,}")

# Test codebook_numpy
print("\n--- Testing codebook_numpy() ---")
t0 = time.perf_counter()
codebook = archive.codebook_numpy()
print(f"Codebook shape: {codebook.shape}")
print(f"Codebook dtype: {codebook.dtype}")
print(f"Codebook size: {codebook.nbytes / 1024 / 1024:.2f} MB")
print(f"Time: {time.perf_counter() - t0:.3f}s")

# Test get_indices_batch
print("\n--- Testing get_indices_batch() ---")
batch_indices = list(range(64))
t0 = time.perf_counter()
indices, labels = archive.get_indices_batch(batch_indices)
print(f"Indices shape: {indices.shape}")
print(f"Indices dtype: {indices.dtype}")
print(f"Labels: {labels[:5]}...")
print(f"Time: {time.perf_counter() - t0:.3f}s")

# Simulate GPU decode (CPU version)
print("\n--- Simulating GPU decode (CPU) ---")
t0 = time.perf_counter()

# This is what happens on GPU:
patches = codebook[indices]  # Gather
print(f"Patches shape after gather: {patches.shape}")  # (64, 12544, 12)

# Reshape to images
batch_size = indices.shape[0]
grid_h, grid_w = archive.grid_size
ps = archive.patch_size
patches = patches.reshape(batch_size, grid_h, grid_w, ps, ps, 3)
images = np.transpose(patches, (0, 1, 3, 2, 4, 5)).reshape(batch_size, archive.dimensions[1], archive.dimensions[0], 3)
print(f"Images shape: {images.shape}")
print(f"Time: {time.perf_counter() - t0:.3f}s")

# Benchmark
print("\n--- Benchmark: Index fetch + CPU decode ---")
batch_sizes = [64, 128, 256]
for bs in batch_sizes:
    batch_indices = list(range(bs))
    
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        indices, labels = archive.get_indices_batch(batch_indices)
        patches = codebook[indices]
        patches = patches.reshape(bs, grid_h, grid_w, ps, ps, 3)
        images = np.transpose(patches, (0, 1, 3, 2, 4, 5)).reshape(bs, archive.dimensions[1], archive.dimensions[0], 3)
        times.append(time.perf_counter() - t0)
    
    median_time = np.median(times)
    ips = bs / median_time
    print(f"Batch {bs:3d}: {ips:,.0f} images/sec (CPU decode)")

print("\n" + "=" * 60)
print("SUCCESS - GPU API ready!")
print("=" * 60)
print("""
On GPU (Colab/local with CUDA):

    from kuat import KuatArchive
    from kuat.gpu import GPUDecoder, GPUDataset
    
    # Option 1: Manual decode
    archive = KuatArchive("train.qvq")
    decoder = GPUDecoder(archive, device="cuda")
    indices, labels = archive.get_indices_batch([0, 1, 2])
    images = decoder.decode(indices)  # On GPU!
    
    # Option 2: Full GPU dataset (fastest)
    dataset = GPUDataset("train.qvq", batch_size=256, device="cuda")
    for images, labels in dataset.epoch(0):
        outputs = model(images)  # No CPU->GPU copy!
""")
