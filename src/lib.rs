//! Kuat Python Bindings
//!
//! High-performance ML dataset loading with O(1) random access.
//!
//! This module provides:
//! - KuatArchive: Read compressed dataset archives
//! - KuatDataset: PyTorch-compatible DataLoader
//!
//! Dataset conversion is done via the `kuat-convert` CLI binary.

use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyValueError, PyRuntimeError, PyIndexError};
use pyo3::types::{PyBytes, PyDict};
use std::sync::Arc;

// Re-use core library types
use quat_tree::grid_vq::{
    MappedGridVQArchive, MemoryLayout, DecodedBatch, QvqDataLoader,
};

// ============================================================================
// KuatArchive - Main archive interface
// ============================================================================

/// A Kuat archive with O(1) random access to image samples.
///
/// Usage:
///     archive = KuatArchive("dataset.qvq")
///     print(len(archive))  # Number of samples
///     print(archive.info())
#[pyclass]
pub struct KuatArchive {
    inner: MappedGridVQArchive,
    #[pyo3(get)]
    path: String,
}

#[pymethods]
impl KuatArchive {
    /// Open an archive file
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        // Use decode_only to skip spatial index (faster for large codebooks)
        let archive = MappedGridVQArchive::open_decode_only(path)
            .map_err(|e| PyIOError::new_err(format!("Failed to open archive: {}", e)))?;
        Ok(Self {
            inner: archive,
            path: path.to_string(),
        })
    }

    /// Number of samples in the archive
    fn __len__(&self) -> usize {
        self.inner.sample_count()
    }

    /// Get archive info as dict
    fn info(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let (w, h) = self.inner.image_dimensions();
        dict.set_item("path", &self.path)?;
        dict.set_item("samples", self.inner.sample_count())?;
        dict.set_item("width", w)?;
        dict.set_item("height", h)?;
        dict.set_item("bits", self.inner.config().bits)?;
        dict.set_item("classes", self.inner.class_names().to_vec())?;
        Ok(dict.into())
    }

    /// Get class names
    #[getter]
    fn classes(&self) -> Vec<String> {
        self.inner.class_names().to_vec()
    }

    /// Decode a single image by index
    /// Returns (image, label) where image is (H, W, 3) numpy array
    fn decode(&self, py: Python<'_>, index: usize) -> PyResult<(PyObject, Option<u32>)> {
        if index >= self.inner.sample_count() {
            return Err(PyIndexError::new_err(format!(
                "Index {} out of range ({})", index, self.inner.sample_count()
            )));
        }

        let pixels = self.inner.decode(index)
            .map_err(|e| PyRuntimeError::new_err(format!("Decode failed: {}", e)))?;

        let label = self.inner.get_label(index).ok();

        let numpy = py.import("numpy")?;
        let (w, h) = self.inner.image_dimensions();
        let shape = (h, w, 3usize);
        let array = numpy.call_method1("frombuffer", (PyBytes::new(py, &pixels), "uint8"))?;
        let reshaped = array.call_method1("reshape", (shape,))?;

        Ok((reshaped.into(), label))
    }

    // ========================================================================
    // GPU Decode Support - expose codebook and indices for GPU gather
    // ========================================================================

    /// Get codebook size (actual number of entries, not max)
    #[getter]
    fn codebook_size(&self) -> usize {
        self.inner.codec().codebook_len()
    }

    /// Get patch size (typically 2 for 2x2 patches)
    #[getter]
    fn patch_size(&self) -> u8 {
        self.inner.config().patch_size
    }

    /// Get image dimensions (width, height)
    #[getter]
    fn dimensions(&self) -> (usize, usize) {
        self.inner.image_dimensions()
    }

    /// Get grid dimensions (patches per row, patches per column)
    #[getter]
    fn grid_size(&self) -> (usize, usize) {
        self.inner.grid_dimensions()
    }

    /// Get codebook as numpy array for GPU decode.
    ///
    /// Returns:
    ///     (K, patch_dim) numpy array with uint8 values [0, 255]
    ///     For 2x2 RGB patches: (codebook_size, 12)
    ///
    /// Usage:
    ///     codebook = archive.codebook_numpy()
    ///     # Upload to GPU once
    ///     codebook_gpu = torch.from_numpy(codebook).cuda()
    fn codebook_numpy(&self, py: Python<'_>) -> PyResult<PyObject> {
        let k = self.inner.codec().codebook_len();
        let ps = self.inner.config().patch_size as usize;
        let patch_dim = ps * ps * 3;

        // Get raw flat codebook bytes
        let bytes = self.inner.codec().flat_codebook();

        let numpy = py.import("numpy")?;
        let array = numpy.call_method1(
            "frombuffer",
            (PyBytes::new(py, bytes), "uint8")
        )?;
        // Make a copy since frombuffer creates a read-only view
        let array_copy = array.call_method0("copy")?;
        let reshaped = array_copy.call_method1("reshape", ((k, patch_dim),))?;

        Ok(reshaped.into())
    }

    /// Get raw VQ indices for a batch of samples (for GPU decode).
    ///
    /// Args:
    ///     indices: List of sample indices to fetch
    ///
    /// Returns:
    ///     (indices_array, labels) where:
    ///     - indices_array: (B, grid_h * grid_w) numpy array of uint32
    ///     - labels: list of labels (or None)
    ///
    /// Usage:
    ///     indices, labels = archive.get_indices_batch([0, 1, 2, 3])
    ///     # GPU decode: images = codebook_gpu[indices]
    fn get_indices_batch(
        &self,
        py: Python<'_>,
        indices: Vec<usize>,
    ) -> PyResult<(PyObject, Vec<Option<u32>>)> {
        // Validate indices
        let sample_count = self.inner.sample_count();
        for &idx in &indices {
            if idx >= sample_count {
                return Err(PyIndexError::new_err(format!(
                    "Index {} out of range ({})", idx, sample_count
                )));
            }
        }

        let mut all_indices = Vec::new();
        let mut labels = Vec::new();

        for &idx in &indices {
            let sample_indices = self.inner.get_indices(idx)
                .map_err(|e| PyRuntimeError::new_err(format!("Get indices failed: {}", e)))?;
            all_indices.extend_from_slice(&sample_indices);
            labels.push(self.inner.get_label(idx).ok());
        }

        // Convert to numpy
        let numpy = py.import("numpy")?;
        let (grid_w, grid_h) = self.inner.grid_dimensions();
        let patches_per_image = grid_h * grid_w;
        let shape = (indices.len(), patches_per_image);

        // Convert u32 indices to bytes
        let bytes: Vec<u8> = all_indices.iter()
            .flat_map(|&i| i.to_le_bytes())
            .collect();

        let array = numpy.call_method1(
            "frombuffer",
            (PyBytes::new(py, &bytes), "uint32")
        )?;
        let array_copy = array.call_method0("copy")?;
        let reshaped = array_copy.call_method1("reshape", (shape,))?;

        Ok((reshaped.into(), labels))
    }

    /// Get all labels as a list
    fn all_labels(&self) -> Vec<Option<u32>> {
        (0..self.inner.sample_count())
            .map(|i| self.inner.get_label(i).ok())
            .collect()
    }

    fn __repr__(&self) -> String {
        let (w, h) = self.inner.image_dimensions();
        format!(
            "KuatArchive('{}', samples={}, {}x{})",
            self.path,
            self.inner.sample_count(),
            w,
            h,
        )
    }
}

// ============================================================================
// KuatDataset - PyTorch-compatible dataset
// ============================================================================

/// PyTorch-compatible dataset for Kuat archives.
///
/// Usage:
///     dataset = KuatDataset("train.kuat", batch_size=64)
///     for epoch in range(10):
///         for batch in dataset.epoch(epoch):
///             images = batch["images"]  # (B, H, W, C) uint8
///             labels = batch["labels"]  # (B,) int32
#[pyclass]
pub struct KuatDataset {
    path: String,
    cached_archive: Arc<MappedGridVQArchive>,
    batch_size: usize,
    shuffle: bool,
    seed: u64,
    drop_last: bool,
    layout: MemoryLayout,
}

#[pymethods]
impl KuatDataset {
    #[new]
    #[pyo3(signature = (path, batch_size=64, shuffle=true, seed=42, drop_last=false, layout="NHWC"))]
    fn new(
        path: &str,
        batch_size: usize,
        shuffle: bool,
        seed: u64,
        drop_last: bool,
        layout: &str,
    ) -> PyResult<Self> {
        // Use decode_only - no spatial index needed for training/inference
        let archive = MappedGridVQArchive::open_decode_only(path)
            .map_err(|e| PyIOError::new_err(format!("Failed to open archive: {}", e)))?;
        
        let layout = match layout.to_uppercase().as_str() {
            "NHWC" => MemoryLayout::NHWC,
            "NCHW" => MemoryLayout::NCHW,
            _ => return Err(PyValueError::new_err("layout must be 'NHWC' or 'NCHW'")),
        };
        
        Ok(Self {
            path: path.to_string(),
            cached_archive: Arc::new(archive),
            batch_size,
            shuffle,
            seed,
            drop_last,
            layout,
        })
    }

    /// Get an iterator for a specific epoch
    fn epoch(&self, epoch: u64) -> PyResult<KuatEpochIterator> {
        let total = self.cached_archive.sample_count();
        let batch_count = if self.drop_last {
            total / self.batch_size
        } else {
            (total + self.batch_size - 1) / self.batch_size
        };
        
        Ok(KuatEpochIterator {
            cached_archive: Arc::clone(&self.cached_archive),
            batch_size: self.batch_size,
            shuffle: self.shuffle,
            seed: self.seed,
            drop_last: self.drop_last,
            layout: self.layout,
            batch_count,
            current_batch: 0,
            epoch,
        })
    }

    /// Number of samples
    fn __len__(&self) -> usize {
        self.cached_archive.sample_count()
    }

    /// Number of batches per epoch
    fn batch_count(&self) -> usize {
        let total = self.cached_archive.sample_count();
        if self.drop_last {
            total / self.batch_size
        } else {
            (total + self.batch_size - 1) / self.batch_size
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "KuatDataset('{}', samples={}, batch_size={})",
            self.path,
            self.cached_archive.sample_count(),
            self.batch_size,
        )
    }
}

// ============================================================================
// Epoch iterator
// ============================================================================

#[pyclass]
pub struct KuatEpochIterator {
    cached_archive: Arc<MappedGridVQArchive>,
    batch_size: usize,
    shuffle: bool,
    seed: u64,
    drop_last: bool,
    layout: MemoryLayout,
    batch_count: usize,
    current_batch: usize,
    epoch: u64,
}

#[pymethods]
impl KuatEpochIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if self.current_batch >= self.batch_count {
            return Ok(None);
        }

        let mut loader = QvqDataLoader::from_shared_archive(Arc::clone(&self.cached_archive))
            .batch_size(self.batch_size)
            .shuffle(self.shuffle)
            .seed(self.seed)
            .drop_last(self.drop_last)
            .layout(self.layout);

        let batch = loader.get_batch(self.epoch, self.current_batch)
            .map_err(|e| PyRuntimeError::new_err(format!("get_batch failed: {}", e)))?;

        self.current_batch += 1;
        Ok(Some(batch_to_dict(py, &batch)?))
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn batch_to_dict(py: Python<'_>, batch: &DecodedBatch) -> PyResult<PyObject> {
    let numpy = py.import("numpy")?;

    let (b, h, w, c) = (batch.batch_size, batch.height, batch.width, batch.channels);
    let shape = match batch.layout {
        MemoryLayout::NHWC => (b, h, w, c),
        MemoryLayout::NCHW => (b, c, h, w),
    };

    // Create array from buffer and make a writable copy
    let array = numpy.call_method1("frombuffer", (PyBytes::new(py, &batch.pixels), "uint8"))?;
    let array_copy = array.call_method0("copy")?;
    let images = array_copy.call_method1("reshape", (shape,))?;

    // Extract labels
    let labels: Vec<u32> = batch.labels.iter().map(|opt| opt.unwrap_or(0)).collect();

    let dict = PyDict::new(py);
    dict.set_item("images", images)?;
    dict.set_item("labels", labels)?;

    // Add captions if present
    if let Some(ref captions) = batch.captions {
        let caption_strings: Vec<&str> = captions
            .iter()
            .map(|opt| opt.as_deref().unwrap_or(""))
            .collect();
        dict.set_item("captions", caption_strings)?;
    }

    Ok(dict.into())
}

// ============================================================================
// Module Definition
// ============================================================================

/// Kuat - Ultra-fast ML dataset loading
#[pymodule]
fn _core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Main classes
    m.add_class::<KuatArchive>()?;
    m.add_class::<KuatDataset>()?;
    m.add_class::<KuatEpochIterator>()?;

    // Version
    m.add("__version__", "0.1.0-beta.1")?;

    Ok(())
}
