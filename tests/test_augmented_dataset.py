"""Unit tests for AugmentedDataset."""

import sys
from pathlib import Path

import pytest
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import AugmentedDataset  # noqa: E402
from config import DataAugmentationConfig  # noqa: E402


class MockDataset:
    """Mock dataset for testing augmentation."""
    
    def __init__(self, size=10, byte_length=1024, pe_dim=256, stat_dim=49):
        self.size = size
        self.byte_length = byte_length
        self.pe_dim = pe_dim
        self.stat_dim = stat_dim
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Return deterministic data for reproducibility
        torch.manual_seed(idx)
        byte_seq = torch.randint(0, 256, (self.byte_length,), dtype=torch.long)
        pe_features = torch.randn(self.pe_dim)
        stat_features = torch.randn(self.stat_dim)
        label = torch.tensor(idx % 2, dtype=torch.long)
        return byte_seq, pe_features, stat_features, label


class TestAugmentedDataset:
    """Test suite for AugmentedDataset."""
    
    def test_disabled_augmentation_passthrough(self):
        """When enable=False, should return original data unchanged."""
        base = MockDataset()
        config = DataAugmentationConfig(enable=False)
        aug = AugmentedDataset(base, config)
        
        # Check length
        assert len(aug) == len(base)
        
        # Check data is unchanged
        for i in range(3):
            orig = base[i]
            augmented = aug[i]
            assert len(orig) == len(augmented)
            assert torch.equal(orig[0], augmented[0])  # byte_seq
            assert torch.equal(orig[1], augmented[1])  # pe_features
            assert torch.equal(orig[2], augmented[2])  # stat_features
            assert torch.equal(orig[3], augmented[3])  # label
    
    def test_byte_dropout(self):
        """byte_dropout should randomly set bytes to zero."""
        base = MockDataset(byte_length=1000)
        config = DataAugmentationConfig(enable=True, byte_dropout=0.1)
        aug = AugmentedDataset(base, config)
        
        # Get original and augmented
        orig = base[0]
        augmented = aug[0]
        
        # Check that some bytes are zeroed
        orig_bytes = orig[0]
        aug_bytes = augmented[0]
        
        # Count zeros
        orig_zeros = (orig_bytes == 0).sum().item()
        aug_zeros = (aug_bytes == 0).sum().item()
        
        # Augmented should have more zeros (with high probability)
        assert aug_zeros >= orig_zeros, "byte_dropout should increase zero count"
        
        # Check that approximately 10% are zeroed
        zeroed = (aug_bytes == 0) & (orig_bytes != 0)
        zero_rate = zeroed.sum().item() / len(orig_bytes)
        assert 0.05 <= zero_rate <= 0.15, f"Expected ~10% dropout, got {zero_rate:.2%}"
    
    def test_byte_noise(self):
        """byte_noise should randomly replace byte values."""
        base = MockDataset(byte_length=1000)
        config = DataAugmentationConfig(enable=True, byte_noise=0.1)
        aug = AugmentedDataset(base, config)
        
        # Get original and augmented
        orig = base[0]
        augmented = aug[0]
        
        # Check that some bytes are changed
        orig_bytes = orig[0]
        aug_bytes = augmented[0]
        
        # Count changes
        changed = (orig_bytes != aug_bytes).sum().item()
        change_rate = changed / len(orig_bytes)
        
        # Should have approximately 10% changes
        assert 0.05 <= change_rate <= 0.15, f"Expected ~10% noise, got {change_rate:.2%}"
    
    def test_feature_noise(self):
        """feature_noise should add Gaussian noise to PE features."""
        base = MockDataset(pe_dim=256)
        config = DataAugmentationConfig(enable=True, feature_noise=0.1)
        aug = AugmentedDataset(base, config)
        
        # Get original and augmented
        orig = base[0]
        augmented = aug[0]
        
        # Check that PE features are modified
        orig_pe = orig[1]
        aug_pe = augmented[1]
        
        # Calculate difference
        diff = (aug_pe - orig_pe).abs()
        
        # Should have small noise (std=0.1)
        mean_diff = diff.mean().item()
        std_diff = diff.std().item()
        
        assert mean_diff < 0.15, f"Mean difference too large: {mean_diff}"
        assert 0.05 <= std_diff <= 0.2, f"Expected std ~0.1, got {std_diff}"
    
    def test_combined_augmentation(self):
        """Multiple augmentations should work together."""
        base = MockDataset()
        config = DataAugmentationConfig(
            enable=True,
            byte_dropout=0.05,
            byte_noise=0.05,
            feature_noise=0.05
        )
        aug = AugmentedDataset(base, config)
        
        # Get original and augmented
        orig = base[0]
        augmented = aug[0]
        
        # Check all components
        assert len(orig) == len(augmented)
        assert not torch.equal(orig[0], augmented[0])  # byte_seq should change
        assert not torch.equal(orig[1], augmented[1])  # pe_features should change
        assert torch.equal(orig[2], augmented[2])  # stat_features unchanged
        assert torch.equal(orig[3], augmented[3])  # label unchanged
    
    def test_original_data_not_modified(self):
        """Augmentation should not modify the original data."""
        base = MockDataset()
        config = DataAugmentationConfig(
            enable=True,
            byte_dropout=0.5,  # High dropout to ensure changes
            byte_noise=0.5,
            feature_noise=0.5
        )
        aug = AugmentedDataset(base, config)
        
        # Get original data
        orig = base[0]
        orig_bytes = orig[0].clone()
        orig_pe = orig[1].clone()
        
        # Apply augmentation multiple times
        for _ in range(5):
            _ = aug[0]
        
        # Check original is unchanged
        assert torch.equal(base[0][0], orig_bytes), "Original byte_seq was modified!"
        assert torch.equal(base[0][1], orig_pe), "Original pe_features was modified!"
    
    def test_deterministic_with_seed(self):
        """Augmentation should be reproducible with same random seed."""
        base = MockDataset()
        config = DataAugmentationConfig(enable=True, byte_dropout=0.1)
        aug = AugmentedDataset(base, config)
        
        # Get augmented data with same seed twice
        torch.manual_seed(42)
        aug1 = aug[0]
        
        torch.manual_seed(42)
        aug2 = aug[0]
        
        # Should be identical
        assert torch.equal(aug1[0], aug2[0]), "Augmentation not deterministic with same seed"
    
    def test_5tuple_passthrough(self):
        """Should handle 5-tuple (with sample_weights) correctly."""
        class MockDatasetWithWeight(MockDataset):
            def __getitem__(self, idx):
                byte_seq, pe_features, stat_features, label = super().__getitem__(idx)
                weight = torch.tensor(1.5, dtype=torch.float32)
                return byte_seq, pe_features, stat_features, label, weight
        
        base = MockDatasetWithWeight()
        config = DataAugmentationConfig(enable=True, byte_dropout=0.1)
        aug = AugmentedDataset(base, config)
        
        # Get augmented
        result = aug[0]
        
        # Should return 5-tuple
        assert len(result) == 5
        byte_seq, pe_features, stat_features, label, weight = result
        assert weight.item() == 1.5, "Weight should be preserved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
