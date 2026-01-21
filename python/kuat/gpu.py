"""
GPU-accelerated VQ decode for Kuat archives.

This module provides PyTorch-based GPU decode, eliminating CPU decode overhead
and reducing PCIe transfer by ~6x (only indices transferred, not pixels).

Usage:
    from kuat import KuatArchive
    from kuat.gpu import GPUDecoder, GPUDataset

    # Method 1: Manual decode
    archive = KuatArchive("train.qvq")
    decoder = GPUDecoder(archive, device="cuda")
    
    indices, labels = archive.get_indices_batch([0, 1, 2, 3])
    images = decoder.decode(indices)  # Already on GPU!

    # Method 2: Full GPU dataset
    dataset = GPUDataset("train.qvq", batch_size=256, device="cuda")
    for images, labels in dataset.epoch(0):
        # images: (B, 3, 224, 224) float32 on GPU
        # labels: (B,) long on GPU
        outputs = model(images)
"""

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class GPUDecoder:
    """
    GPU-accelerated VQ decoder using PyTorch gather.
    
    The codebook is uploaded to GPU once (~768 KB for 16-bit, ~270 MB for 24-bit).
    Decode is just: decoded = codebook[indices] - massively parallel on GPU.
    
    Benefits:
    - Zero CPU decode overhead
    - 6x less PCIe bandwidth (indices vs pixels)
    - Images decoded directly on GPU memory
    """
    
    def __init__(self, archive, device: str = "cuda", dtype=None):
        """
        Initialize GPU decoder from a KuatArchive.
        
        Args:
            archive: KuatArchive instance
            device: PyTorch device ("cuda", "cuda:0", etc.)
            dtype: Output dtype (default: float32 normalized to [0,1])
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for GPU decode. Install with: pip install torch")
        
        self.device = device
        self.dtype = dtype or torch.float32
        
        # Get archive info
        self.width, self.height = archive.dimensions
        self.patch_size = archive.patch_size
        self.grid_w, self.grid_h = archive.grid_size
        self.patches_per_image = self.grid_w * self.grid_h
        
        # Upload codebook to GPU (one-time cost)
        codebook_np = archive.codebook_numpy()  # (K, 12)
        self.codebook = torch.from_numpy(codebook_np).to(device)
        
        # Pre-convert to float and normalize if using float output
        if self.dtype in (torch.float32, torch.float16, torch.bfloat16):
            self.codebook = self.codebook.to(self.dtype) / 255.0
        
        self._codebook_mb = self.codebook.numel() * self.codebook.element_size() / 1e6
    
    def decode(self, indices: np.ndarray, layout: str = "NCHW", 
               normalize: bool = False, mean=None, std=None) -> "torch.Tensor":
        """
        Decode VQ indices to images on GPU.
        
        Args:
            indices: (B, patches) numpy array of uint32 indices
            layout: "NCHW" (default, for PyTorch) or "NHWC"
            normalize: Apply ImageNet normalization
            mean: Custom mean for normalization (default: ImageNet)
            std: Custom std for normalization (default: ImageNet)
            
        Returns:
            torch.Tensor on GPU:
            - NCHW: (B, 3, H, W) 
            - NHWC: (B, H, W, 3)
        """
        batch_size = indices.shape[0]
        
        # Transfer indices to GPU (small: ~50 KB for batch of 64)
        indices_gpu = torch.from_numpy(indices.astype(np.int64)).to(self.device)
        
        # GPU gather - the fast part!
        # (B, patches) -> (B, patches, 12)
        patches = self.codebook[indices_gpu]
        
        # Reshape: (B, grid_h, grid_w, patch_h, patch_w, 3)
        patches = patches.view(
            batch_size, self.grid_h, self.grid_w,
            self.patch_size, self.patch_size, 3
        )
        
        # Rearrange to image: (B, H, W, 3)
        images = patches.permute(0, 1, 3, 2, 4, 5).contiguous()
        images = images.view(batch_size, self.height, self.width, 3)
        
        if layout == "NCHW":
            images = images.permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)
        
        # Optional normalization
        if normalize:
            if mean is None:
                mean = [0.485, 0.456, 0.406]  # ImageNet
            if std is None:
                std = [0.229, 0.224, 0.225]  # ImageNet
            
            mean_t = torch.tensor(mean, device=self.device, dtype=self.dtype)
            std_t = torch.tensor(std, device=self.device, dtype=self.dtype)
            
            if layout == "NCHW":
                mean_t = mean_t.view(1, 3, 1, 1)
                std_t = std_t.view(1, 3, 1, 1)
            else:
                mean_t = mean_t.view(1, 1, 1, 3)
                std_t = std_t.view(1, 1, 1, 3)
            
            images = (images - mean_t) / std_t
        
        return images
    
    def __repr__(self):
        return (f"GPUDecoder(device={self.device}, "
                f"codebook={self.codebook.shape}, "
                f"memory={self._codebook_mb:.1f}MB)")


class GPUDataset:
    """
    Full GPU dataset for training - codebook and indices on GPU.
    
    For maximum performance, pre-loads all indices to GPU memory.
    Decode is just a gather operation with no CPU involvement.
    
    Memory usage:
    - Codebook: ~768 KB (16-bit) to ~270 MB (24-bit)
    - Indices: ~50 MB per 100K images (at 224x224)
    """
    
    def __init__(self, archive_or_path, batch_size: int = 64,
                 device: str = "cuda", preload: bool = True,
                 normalize: bool = True, mean=None, std=None):
        """
        Initialize GPU dataset.
        
        Args:
            archive_or_path: KuatArchive instance or path to .qvq file
            batch_size: Batch size for iteration
            device: PyTorch device
            preload: If True, load all indices to GPU (faster, more memory)
            normalize: Apply normalization during decode
            mean: Normalization mean (default: ImageNet)
            std: Normalization std (default: ImageNet)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for GPU decode")
        
        # Import here to avoid circular import
        from kuat import KuatArchive
        
        if isinstance(archive_or_path, str):
            self.archive = KuatArchive(archive_or_path)
        else:
            self.archive = archive_or_path
        
        self.batch_size = batch_size
        self.device = device
        self.normalize = normalize
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        
        # Create GPU decoder
        self.decoder = GPUDecoder(self.archive, device=device)
        
        # Get dimensions
        self.num_samples = len(self.archive)
        self.width, self.height = self.archive.dimensions
        
        # Pre-load all indices and labels to GPU
        self._indices_gpu = None
        self._labels_gpu = None
        
        if preload:
            self._preload_to_gpu()
        
        # Pre-compute normalization tensors
        self._mean_gpu = torch.tensor(self.mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self._std_gpu = torch.tensor(self.std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    
    def _preload_to_gpu(self):
        """Pre-load all indices and labels to GPU."""
        print(f"Pre-loading {self.num_samples:,} samples to GPU...")
        
        # Batch load to avoid memory spikes
        CHUNK_SIZE = 1000
        all_indices = []
        all_labels = []
        
        for start in range(0, self.num_samples, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, self.num_samples)
            batch_indices = list(range(start, end))
            
            indices, labels = self.archive.get_indices_batch(batch_indices)
            all_indices.append(indices)
            all_labels.extend(labels)
        
        # Concatenate and upload
        indices_np = np.concatenate(all_indices, axis=0)
        labels_np = np.array([l if l is not None else -1 for l in all_labels], dtype=np.int64)
        
        self._indices_gpu = torch.from_numpy(indices_np).to(self.device)
        self._labels_gpu = torch.from_numpy(labels_np).to(self.device)
        
        indices_mb = self._indices_gpu.numel() * self._indices_gpu.element_size() / 1e6
        print(f"  Indices on GPU: {self._indices_gpu.shape}, {indices_mb:.1f} MB")
    
    def __len__(self):
        return self.num_samples
    
    def batch_count(self):
        """Number of batches per epoch."""
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def epoch(self, epoch_num: int, shuffle: bool = True):
        """
        Iterate over one epoch.
        
        Args:
            epoch_num: Epoch number (used for shuffle seed)
            shuffle: Whether to shuffle samples
            
        Yields:
            (images, labels) tuples where:
            - images: (B, 3, H, W) float32 tensor on GPU
            - labels: (B,) long tensor on GPU
        """
        # Generate permutation on GPU
        if shuffle:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(42 + epoch_num)
            perm = torch.randperm(self.num_samples, device=self.device, generator=generator)
        else:
            perm = torch.arange(self.num_samples, device=self.device)
        
        # Iterate batches
        for start in range(0, self.num_samples, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            batch_perm = perm[start:end]
            
            if self._indices_gpu is not None:
                # Fast path: gather from GPU
                batch_indices = self._indices_gpu[batch_perm]
                batch_labels = self._labels_gpu[batch_perm]
                
                # GPU decode (gather)
                patches = self.decoder.codebook[batch_indices.long()]
                
                # Reshape to images
                B = batch_indices.shape[0]
                patches = patches.view(
                    B, self.decoder.grid_h, self.decoder.grid_w,
                    self.decoder.patch_size, self.decoder.patch_size, 3
                )
                images = patches.permute(0, 1, 3, 2, 4, 5).contiguous()
                images = images.view(B, self.height, self.width, 3)
                images = images.permute(0, 3, 1, 2).contiguous()  # NCHW
                
                # Normalize
                if self.normalize:
                    images = (images - self._mean_gpu) / self._std_gpu
                
                yield images, batch_labels
            else:
                # Slow path: load from archive
                sample_indices = batch_perm.cpu().tolist()
                indices, labels = self.archive.get_indices_batch(sample_indices)
                
                images = self.decoder.decode(
                    indices, layout="NCHW", 
                    normalize=self.normalize, 
                    mean=self.mean, std=self.std
                )
                labels_t = torch.tensor(
                    [l if l is not None else -1 for l in labels],
                    device=self.device, dtype=torch.long
                )
                
                yield images, labels_t
    
    def __repr__(self):
        return (f"GPUDataset(samples={self.num_samples:,}, "
                f"batch_size={self.batch_size}, device={self.device})")
