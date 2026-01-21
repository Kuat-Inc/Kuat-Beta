#!/usr/bin/env python3
"""Benchmark: Kuat vs PyTorch ImageFolder DataLoader"""

import time
import os
import sys

# ============================================================================
# Kuat Benchmark
# ============================================================================
def bench_kuat(archive_path, batch_size=64, num_batches=100):
    from kuat import KuatDataset
    
    dataset = KuatDataset(archive_path, batch_size=batch_size, shuffle=True)
    
    # Warmup
    for i, batch in enumerate(dataset.epoch(0)):
        if i >= 5:
            break
    
    # Timed run
    start = time.perf_counter()
    total_images = 0
    for i, batch in enumerate(dataset.epoch(1)):
        total_images += batch['images'].shape[0]
        if i >= num_batches:
            break
    elapsed = time.perf_counter() - start
    
    return total_images, elapsed

# ============================================================================
# PyTorch ImageFolder Benchmark
# ============================================================================
def bench_pytorch(folder_path, batch_size=64, num_batches=100, num_workers=4):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(folder_path, transform=transform)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    # Warmup
    loader_iter = iter(loader)
    for _ in range(5):
        try:
            next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
    
    # Timed run
    start = time.perf_counter()
    total_images = 0
    loader_iter = iter(loader)
    for i in range(num_batches + 1):
        try:
            images, labels = next(loader_iter)
            total_images += images.shape[0]
        except StopIteration:
            break
    elapsed = time.perf_counter() - start
    
    return total_images, elapsed

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    # Paths
    qvq_path = r"C:\Users\yaweh\OneDrive\Documents\Kuat Inc Code\Quattree\imagewoof_4bit_adaptive_33m.qvq"
    # ImageWoof extracted folder (if available)
    imagewoof_folder = r"C:\Users\yaweh\Downloads\imagewoof2\imagewoof2\train"
    
    batch_size = 64
    num_batches = 100
    
    print("=" * 60)
    print("KUAT vs PyTorch ImageFolder Benchmark")
    print("=" * 60)
    print(f"Batch size: {batch_size}")
    print(f"Batches: {num_batches}")
    print()
    
    # Kuat benchmark
    print("Running Kuat benchmark...")
    kuat_images, kuat_time = bench_kuat(qvq_path, batch_size, num_batches)
    kuat_ips = kuat_images / kuat_time
    print(f"  Kuat: {kuat_images:,} images in {kuat_time:.2f}s = {kuat_ips:,.0f} images/sec")
    print()
    
    # PyTorch benchmark (only if folder exists)
    if os.path.exists(imagewoof_folder):
        for num_workers in [0, 2, 4, 8]:
            print(f"Running PyTorch benchmark (workers={num_workers})...")
            try:
                pt_images, pt_time = bench_pytorch(imagewoof_folder, batch_size, num_batches, num_workers)
                pt_ips = pt_images / pt_time
                print(f"  PyTorch (workers={num_workers}): {pt_images:,} images in {pt_time:.2f}s = {pt_ips:,.0f} images/sec")
                print(f"  Kuat speedup: {kuat_ips / pt_ips:.1f}x faster")
            except Exception as e:
                print(f"  Error: {e}")
            print()
    else:
        print(f"PyTorch benchmark skipped - folder not found: {imagewoof_folder}")
        print("To compare, extract ImageWoof to that path.")
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Kuat throughput: {kuat_ips:,.0f} images/sec")
    print(f"At 224x224 RGB: {kuat_ips * 224 * 224 * 3 / 1e9:.2f} GB/sec of decoded pixels")
