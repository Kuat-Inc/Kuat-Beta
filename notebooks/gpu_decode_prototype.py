# %% [markdown]
# # Kuat GPU Decode - Colab Prototype
# 
# This notebook tests GPU-accelerated VQ decode for Kuat archives.
# The decode operation is a simple gather (table lookup) which GPUs excel at.
#
# ## Setup
# 1. Upload a .qvq file to Colab
# 2. Run this notebook
# 3. Compare CPU vs GPU decode speed

# %% 
# Install dependencies
# !pip install numpy

# %%
import numpy as np
import time

# Check for GPU
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("✗ No CUDA - will simulate with CPU tensors")
except ImportError:
    HAS_CUDA = False
    print("✗ PyTorch not installed")

# %%
# =============================================================================
# GPU Codebook Decoder
# =============================================================================

class GPUCodebookDecoder:
    """
    GPU-accelerated VQ decode using PyTorch gather operations.
    
    The codebook is uploaded to GPU once, then decode is just:
        decoded_pixels = codebook[indices]
    
    This is ~100x faster than CPU decode for large batches.
    """
    
    def __init__(self, codebook_np: np.ndarray, device: str = "cuda"):
        """
        Args:
            codebook_np: (num_entries, patch_size) uint8 array
                         For 2x2 RGB patches: (65536, 12)
            device: "cuda" or "cpu"
        """
        self.device = device if HAS_CUDA else "cpu"
        self.num_entries = codebook_np.shape[0]
        self.patch_size = codebook_np.shape[1]
        
        # Upload codebook to GPU (one-time cost)
        # Keep as uint8 to save memory, convert during gather
        self.codebook = torch.from_numpy(codebook_np).to(self.device)
        
        print(f"Codebook on {self.device}: {self.codebook.shape}, "
              f"{self.codebook.numel() * 1 / 1e6:.2f} MB")
    
    def decode_batch(self, indices: np.ndarray, width: int, height: int,
                     layout: str = "NHWC") -> torch.Tensor:
        """
        Decode a batch of images from VQ indices.
        
        Args:
            indices: (batch, patches_h * patches_w) uint16/uint32 array
            width: image width in pixels
            height: image height in pixels  
            layout: "NHWC" or "NCHW"
            
        Returns:
            Decoded images as torch tensor on GPU
            NHWC: (batch, height, width, 3) uint8
            NCHW: (batch, 3, height, width) uint8
        """
        batch_size = indices.shape[0]
        patch_h, patch_w = 2, 2  # Grid VQ uses 2x2 patches
        patches_h = height // patch_h
        patches_w = width // patch_w
        
        # Send indices to GPU (small transfer - just 2-4 bytes per patch)
        indices_gpu = torch.from_numpy(indices.astype(np.int64)).to(self.device)
        
        # GPU gather - this is the fast part!
        # Shape: (batch, patches, 12) where 12 = 2x2x3
        patches = self.codebook[indices_gpu]  # (B, P, 12)
        
        # Reshape to image
        # (B, patches_h, patches_w, patch_h, patch_w, 3)
        patches = patches.reshape(batch_size, patches_h, patches_w, patch_h, patch_w, 3)
        
        # Transpose to get pixels in right order
        # (B, patches_h, patch_h, patches_w, patch_w, 3) -> (B, H, W, 3)
        images = patches.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, height, width, 3)
        
        if layout == "NCHW":
            images = images.permute(0, 3, 1, 2)  # (B, 3, H, W)
        
        return images
    
    def decode_batch_float(self, indices: np.ndarray, width: int, height: int,
                           layout: str = "NCHW", normalize: bool = True) -> torch.Tensor:
        """
        Decode and convert to float32, ready for model input.
        
        Args:
            normalize: If True, scale to [0, 1] range
            
        Returns:
            (batch, 3, height, width) float32 tensor on GPU
        """
        images = self.decode_batch(indices, width, height, layout)
        images = images.float()
        if normalize:
            images = images / 255.0
        return images


# %%
# =============================================================================
# Simulated QVQ Reader (for testing without full Kuat)
# =============================================================================

def create_test_data(num_images: int = 1000, width: int = 224, height: int = 224,
                     codebook_bits: int = 16):
    """
    Create simulated QVQ data for testing GPU decode.
    
    In production, this would come from the actual .qvq file reader.
    """
    patch_h, patch_w = 2, 2
    patches_h = height // patch_h
    patches_w = width // patch_w
    patches_per_image = patches_h * patches_w
    
    # Simulated codebook (would be loaded from .qvq header)
    codebook_size = min(2 ** codebook_bits, 65536)
    codebook = np.random.randint(0, 256, (codebook_size, 12), dtype=np.uint8)
    
    # Simulated indices (would come from .qvq sample data)
    indices = np.random.randint(0, codebook_size, (num_images, patches_per_image), 
                                dtype=np.uint16)
    
    return codebook, indices, width, height


# %%
# =============================================================================
# Benchmark: CPU vs GPU Decode
# =============================================================================

def benchmark_cpu_decode(codebook: np.ndarray, indices: np.ndarray, 
                         width: int, height: int, num_iterations: int = 10):
    """Baseline: NumPy CPU decode"""
    batch_size = indices.shape[0]
    patch_h, patch_w = 2, 2
    patches_h = height // patch_h
    patches_w = width // patch_w
    
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        
        # CPU gather
        patches = codebook[indices]  # (B, P, 12)
        patches = patches.reshape(batch_size, patches_h, patches_w, patch_h, patch_w, 3)
        images = np.transpose(patches, (0, 1, 3, 2, 4, 5)).reshape(batch_size, height, width, 3)
        
        times.append(time.perf_counter() - start)
    
    return np.median(times), images


def benchmark_gpu_decode(decoder: GPUCodebookDecoder, indices: np.ndarray,
                         width: int, height: int, num_iterations: int = 10):
    """GPU decode using PyTorch gather"""
    times = []
    
    # Warmup
    for _ in range(3):
        _ = decoder.decode_batch(indices, width, height)
        if HAS_CUDA:
            torch.cuda.synchronize()
    
    for _ in range(num_iterations):
        start = time.perf_counter()
        
        images = decoder.decode_batch(indices, width, height)
        if HAS_CUDA:
            torch.cuda.synchronize()
        
        times.append(time.perf_counter() - start)
    
    return np.median(times), images


# %%
# =============================================================================
# Run Benchmark
# =============================================================================

print("=" * 60)
print("GPU VQ Decode Benchmark")
print("=" * 60)

# Test parameters
batch_sizes = [32, 64, 128, 256, 512]
width, height = 224, 224

# Create codebook (simulated - in production loaded from .qvq)
codebook_size = 65536  # 16-bit indices
codebook = np.random.randint(0, 256, (codebook_size, 12), dtype=np.uint8)

# Initialize GPU decoder
decoder = GPUCodebookDecoder(codebook)

print(f"\nImage size: {width}x{height}")
print(f"Patches per image: {(width//2) * (height//2)} = {width*height//4}")
print()

results = []
for batch_size in batch_sizes:
    # Create test indices
    patches_per_image = (width // 2) * (height // 2)
    indices = np.random.randint(0, codebook_size, (batch_size, patches_per_image), 
                                dtype=np.uint16)
    
    # CPU benchmark
    cpu_time, cpu_images = benchmark_cpu_decode(codebook, indices, width, height)
    cpu_ips = batch_size / cpu_time
    
    # GPU benchmark
    gpu_time, gpu_images = benchmark_gpu_decode(decoder, indices, width, height)
    gpu_ips = batch_size / gpu_time
    
    speedup = gpu_ips / cpu_ips
    
    print(f"Batch {batch_size:3d}: CPU {cpu_ips:6,.0f} img/s | GPU {gpu_ips:8,.0f} img/s | {speedup:5.1f}x faster")
    results.append((batch_size, cpu_ips, gpu_ips, speedup))

# %%
# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

best = max(results, key=lambda x: x[2])
print(f"\nBest GPU throughput: {best[2]:,.0f} images/sec (batch_size={best[0]})")
print(f"Best speedup over CPU: {best[3]:.1f}x")

if HAS_CUDA:
    # Memory analysis
    indices_per_image = (width // 2) * (height // 2)
    bytes_per_image_indices = indices_per_image * 2  # uint16
    bytes_per_image_decoded = width * height * 3  # RGB uint8
    
    print(f"\nMemory per image:")
    print(f"  Indices (to GPU): {bytes_per_image_indices:,} bytes ({bytes_per_image_indices/1024:.1f} KB)")
    print(f"  Decoded (on GPU): {bytes_per_image_decoded:,} bytes ({bytes_per_image_decoded/1024:.1f} KB)")
    print(f"  Ratio: {bytes_per_image_decoded / bytes_per_image_indices:.1f}x expansion")
    
    print(f"\nPCIe bandwidth savings:")
    print(f"  Traditional: Copy {bytes_per_image_decoded/1024:.0f} KB decoded pixels per image")
    print(f"  GPU decode:  Copy {bytes_per_image_indices/1024:.1f} KB indices per image")
    print(f"  Savings: {(1 - bytes_per_image_indices/bytes_per_image_decoded)*100:.0f}% less PCIe traffic")

# %%
# =============================================================================
# Integration Example
# =============================================================================

print("\n" + "=" * 60)
print("INTEGRATION EXAMPLE")
print("=" * 60)

print("""
# In your training loop:

from kuat import KuatArchive
from kuat.gpu import GPUCodebookDecoder  # New!

# Load archive and create GPU decoder
archive = KuatArchive("imagenet.qvq")
gpu_decoder = GPUCodebookDecoder(archive.codebook(), device="cuda")

# Training loop
for epoch in range(100):
    for indices, labels in archive.iter_indices(batch_size=256):
        # GPU decode - images are already on GPU!
        images = gpu_decoder.decode_batch_float(indices, 224, 224)
        
        # Forward pass - no CPU->GPU copy needed!
        outputs = model(images)
        loss = criterion(outputs, labels.cuda())
        ...
""")
