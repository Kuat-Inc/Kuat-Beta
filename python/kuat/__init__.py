# kuat - Ultra-fast ML dataset loading
#
# 10x faster dataloading through intelligent compression.
# GPU decode: 50-100x faster with minimal PCIe transfer.

__version__ = "0.1.0-beta.1"

from ._core import (
    KuatArchive,
    KuatDataset,
    KuatEpochIterator,
)

# GPU decode (optional - requires PyTorch)
try:
    from .gpu import GPUDecoder, GPUDataset
    __all__ = [
        # Archive access
        "KuatArchive",
        # CPU dataset
        "KuatDataset",
        "KuatEpochIterator",
        # GPU decode
        "GPUDecoder",
        "GPUDataset",
        "__version__",
    ]
except ImportError:
    # PyTorch not installed - GPU decode unavailable
    __all__ = [
        "KuatArchive",
        "KuatDataset",
        "KuatEpochIterator",
        "__version__",
    ]
