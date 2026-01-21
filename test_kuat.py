#!/usr/bin/env python3
"""Test Kuat package functionality"""

from kuat import KuatArchive, KuatDataset
import time

# Test archive
archive_path = r'C:\Users\yaweh\OneDrive\Documents\Kuat Inc Code\Quattree\imagewoof_4bit_adaptive_33m.qvq'
archive = KuatArchive(archive_path)
print(f'Archive: {archive}')
print(f'Info: {archive.info()}')

# Test dataset
dataset = KuatDataset(archive_path, batch_size=32, shuffle=True)
print(f'\nDataset: batch_size=32, len={len(dataset)}')

# Test iteration
start = time.time()
batches = 0
for batch in dataset.epoch(0):
    batches += 1
    if batches == 1:
        print(f"First batch: images shape={batch['images'].shape}, labels={batch['labels'][:5]}...")
    if batches >= 10:
        break
elapsed = time.time() - start
print(f'Loaded {batches} batches in {elapsed:.3f}s ({batches/elapsed:.1f} batches/sec)')
