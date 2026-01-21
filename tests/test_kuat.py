"""Basic tests for kuat package."""

import pytest


def test_import():
    """Test that kuat can be imported."""
    import kuat
    assert hasattr(kuat, '__version__')
    assert hasattr(kuat, 'KuatArchive')
    assert hasattr(kuat, 'KuatDataset')
    assert hasattr(kuat, 'convert_dataset')


def test_version():
    """Test version string."""
    from kuat import __version__
    assert __version__.startswith('0.1.0')


# Integration tests (require actual archive files)
@pytest.mark.skip(reason="Requires test data")
def test_open_archive():
    """Test opening an archive."""
    from kuat import KuatArchive
    archive = KuatArchive("test.kuat")
    assert len(archive) > 0


@pytest.mark.skip(reason="Requires test data")
def test_dataset_iteration():
    """Test dataset iteration."""
    from kuat import KuatDataset
    dataset = KuatDataset("test.kuat", batch_size=32)
    
    batch_count = 0
    for batch in dataset.epoch(0):
        assert "images" in batch
        assert "labels" in batch
        assert batch["images"].shape[0] <= 32
        batch_count += 1
    
    assert batch_count > 0
