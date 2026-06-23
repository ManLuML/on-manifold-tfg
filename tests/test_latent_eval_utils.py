"""Tests for experiment utilities.

Tests cover:
- decode_and_save_images: VAE decode + save pipeline
- count_existing_images: Resume functionality
- get_max_existing_index: Max index detection
- get_generation_tasks: Distributed generation task allocation
- get_class_labels_for_balanced_generation: Class-balanced label generation
- save_single_image: Single image save with class prefix
- count_images_per_class: Per-class image counting for HP search
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path for experiments module import
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import (
    count_existing_images,
    count_images_per_class,
    decode_and_save_images,
    get_class_labels_for_balanced_generation,
    get_generation_tasks,
    get_max_existing_index,
    save_single_image,
)

# =============================================================================
# Mock Classes
# =============================================================================


class MockVAE:
    """Mock VAE handler for testing decode_and_save_images."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Return deterministic images based on input for reproducibility.

        Maps latent tensor to [-1, 1] range images.
        """
        batch_size = z.shape[0]
        # Create deterministic output based on input mean
        mean_val = z.mean().item()
        # Generate images in [-1, 1] range
        images = torch.full((batch_size, 3, 256, 256), mean_val, device=z.device)
        images = torch.clamp(images, -1.0, 1.0)
        return images


class RandomMockVAE:
    """Mock VAE that returns random images in [-1, 1] range."""

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        return torch.rand(batch_size, 3, 256, 256, device=z.device) * 2 - 1


# =============================================================================
# decode_and_save_images Tests
# =============================================================================


class TestDecodeAndSaveImages:
    """Tests for VAE decode + save pipeline."""

    @pytest.fixture
    def mock_vae(self) -> MockVAE:
        """Create mock VAE."""
        return MockVAE()

    @pytest.fixture
    def temp_output_dir(self, tmp_path: Path) -> Path:
        """Create temporary output directory."""
        return tmp_path / "output_images"

    def test_decode_and_save_creates_correct_files(self, mock_vae: MockVAE, temp_output_dir: Path) -> None:
        """Images should be saved with correct filenames (zero-padded 5 digits)."""
        latents = torch.randn(4, 4, 32, 32)

        num_saved = decode_and_save_images(
            latents=latents,
            vae=mock_vae,
            output_folder=temp_output_dir,
            start_idx=0,
            decode_batch_size=2,
        )

        assert num_saved == 4
        assert temp_output_dir.exists()

        # Check filenames
        expected_files = ["00000.png", "00001.png", "00002.png", "00003.png"]
        actual_files = sorted([f.name for f in temp_output_dir.glob("*.png")])
        assert actual_files == expected_files

    def test_decode_and_save_with_start_idx(self, mock_vae: MockVAE, temp_output_dir: Path) -> None:
        """start_idx should offset the filenames."""
        latents = torch.randn(3, 4, 32, 32)

        num_saved = decode_and_save_images(
            latents=latents,
            vae=mock_vae,
            output_folder=temp_output_dir,
            start_idx=100,
            decode_batch_size=16,
        )

        assert num_saved == 3
        expected_files = ["00100.png", "00101.png", "00102.png"]
        actual_files = sorted([f.name for f in temp_output_dir.glob("*.png")])
        assert actual_files == expected_files

    def test_decode_batch_size_memory_efficiency(self, mock_vae: MockVAE, temp_output_dir: Path) -> None:
        """Small decode_batch_size should split processing correctly."""
        latents = torch.randn(10, 4, 32, 32)

        # Use small decode batch size
        num_saved = decode_and_save_images(
            latents=latents,
            vae=mock_vae,
            output_folder=temp_output_dir,
            start_idx=0,
            decode_batch_size=3,  # Will process in 4 batches: 3, 3, 3, 1
        )

        assert num_saved == 10
        assert len(list(temp_output_dir.glob("*.png"))) == 10

    def test_image_format_is_png(self, mock_vae: MockVAE, temp_output_dir: Path) -> None:
        """Saved images should be PNG format."""
        latents = torch.randn(1, 4, 32, 32)

        decode_and_save_images(
            latents=latents,
            vae=mock_vae,
            output_folder=temp_output_dir,
            start_idx=0,
            decode_batch_size=16,
        )

        saved_files = list(temp_output_dir.glob("*"))
        assert len(saved_files) == 1
        assert saved_files[0].suffix == ".png"

    def test_creates_output_directory(self, mock_vae: MockVAE, tmp_path: Path) -> None:
        """Should create output directory if it doesn't exist."""
        output_dir = tmp_path / "nested" / "deep" / "output"
        assert not output_dir.exists()

        latents = torch.randn(1, 4, 32, 32)
        decode_and_save_images(
            latents=latents,
            vae=mock_vae,
            output_folder=output_dir,
            start_idx=0,
            decode_batch_size=16,
        )

        assert output_dir.exists()
        assert (output_dir / "00000.png").exists()


# =============================================================================
# count_existing_images Tests
# =============================================================================


class TestCountExistingImages:
    """Tests for resume functionality image counting."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory should return 0."""
        assert count_existing_images(tmp_path) == 0

    def test_counts_only_png_files(self, tmp_path: Path) -> None:
        """Should count only PNG files, not other types."""
        # Create various files
        (tmp_path / "00000.png").touch()
        (tmp_path / "00001.png").touch()
        (tmp_path / "readme.txt").touch()
        (tmp_path / "image.jpg").touch()
        (tmp_path / "data.json").touch()

        assert count_existing_images(tmp_path) == 2

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Nonexistent directory should return 0."""
        nonexistent = tmp_path / "does_not_exist"
        assert not nonexistent.exists()
        assert count_existing_images(nonexistent) == 0

    def test_counts_correct_number(self, tmp_path: Path) -> None:
        """Should count exact number of PNG files."""
        for i in range(100):
            (tmp_path / f"{i:05d}.png").touch()

        assert count_existing_images(tmp_path) == 100


# =============================================================================
# get_max_existing_index Tests
# =============================================================================


class TestGetMaxExistingIndex:
    """Tests for maximum index detection."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory should return -1."""
        assert get_max_existing_index(tmp_path) == -1

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Nonexistent directory should return -1."""
        nonexistent = tmp_path / "does_not_exist"
        assert get_max_existing_index(nonexistent) == -1

    def test_finds_max_index(self, tmp_path: Path) -> None:
        """Should find the maximum index correctly."""
        (tmp_path / "00000.png").touch()
        (tmp_path / "00005.png").touch()
        (tmp_path / "00042.png").touch()
        (tmp_path / "00010.png").touch()

        assert get_max_existing_index(tmp_path) == 42

    def test_ignores_non_numeric_files(self, tmp_path: Path) -> None:
        """Should ignore files that don't match numeric pattern."""
        (tmp_path / "00000.png").touch()
        (tmp_path / "00010.png").touch()
        (tmp_path / "image.png").touch()
        (tmp_path / "test_00999.png").touch()  # Has prefix, should be ignored

        assert get_max_existing_index(tmp_path) == 10

    def test_handles_large_indices(self, tmp_path: Path) -> None:
        """Should handle large index numbers."""
        (tmp_path / "00000.png").touch()
        (tmp_path / "99999.png").touch()

        assert get_max_existing_index(tmp_path) == 99999


# =============================================================================
# get_generation_tasks Tests
# =============================================================================


class TestGetGenerationTasks:
    """Tests for distributed generation task allocation."""

    def test_single_gpu_no_existing(self) -> None:
        """Single GPU with no existing images."""
        start_idx, num_to_generate, num_steps = get_generation_tasks(
            num_images=1000,
            existing_count=0,
            world_size=1,
            rank=0,
            batch_size=100,
        )

        assert start_idx == 0
        assert num_to_generate == 1000
        assert num_steps == 10

    def test_single_gpu_with_existing(self) -> None:
        """Single GPU with some existing images (resume)."""
        start_idx, num_to_generate, num_steps = get_generation_tasks(
            num_images=1000,
            existing_count=500,
            world_size=1,
            rank=0,
            batch_size=100,
        )

        assert start_idx == 500
        assert num_to_generate == 500
        assert num_steps == 5

    def test_all_images_exist(self) -> None:
        """All images already exist."""
        start_idx, num_to_generate, num_steps = get_generation_tasks(
            num_images=1000,
            existing_count=1000,
            world_size=1,
            rank=0,
            batch_size=100,
        )

        assert start_idx == 0
        assert num_to_generate == 0
        assert num_steps == 0

    def test_multi_gpu_even_distribution(self) -> None:
        """Multi-GPU with evenly divisible work."""
        results = []
        for rank in range(4):
            start_idx, num_to_generate, num_steps = get_generation_tasks(
                num_images=1000,
                existing_count=0,
                world_size=4,
                rank=rank,
                batch_size=50,
            )
            results.append((start_idx, num_to_generate, num_steps))

        # Each GPU should generate 250 images
        for _, num_to_generate, _ in results:
            assert num_to_generate == 250

        # Total coverage should be 1000
        total = sum(r[1] for r in results)
        assert total == 1000

    def test_multi_gpu_uneven_distribution(self) -> None:
        """Multi-GPU with remainder (extra images to lower ranks)."""
        results = []
        for rank in range(4):
            start_idx, num_to_generate, num_steps = get_generation_tasks(
                num_images=1002,
                existing_count=0,
                world_size=4,
                rank=rank,
                batch_size=50,
            )
            results.append((start_idx, num_to_generate, num_steps))

        # Ranks 0 and 1 get 251, ranks 2 and 3 get 250
        assert results[0][1] == 251
        assert results[1][1] == 251
        assert results[2][1] == 250
        assert results[3][1] == 250

        # Total should be 1002
        total = sum(r[1] for r in results)
        assert total == 1002

    def test_num_steps_calculation(self) -> None:
        """num_steps should be ceiling division."""
        _, _, num_steps = get_generation_tasks(
            num_images=105,
            existing_count=0,
            world_size=1,
            rank=0,
            batch_size=100,
        )

        # 105 / 100 = 1.05, ceiling = 2
        assert num_steps == 2


# =============================================================================
# get_class_labels_for_balanced_generation Tests
# =============================================================================


class TestGetClassLabelsForBalancedGeneration:
    """Tests for class-balanced label generation."""

    def test_exact_division(self) -> None:
        """Each class should get exactly equal labels when evenly divisible."""
        labels = get_class_labels_for_balanced_generation(num_images=1000, num_classes=1000)

        assert len(labels) == 1000

        # Each class should appear exactly once
        unique, counts = np.unique(labels, return_counts=True)
        assert len(unique) == 1000
        assert all(c == 1 for c in counts)

    def test_balanced_distribution(self) -> None:
        """Distribution should be balanced with equal counts per class."""
        labels = get_class_labels_for_balanced_generation(num_images=10000, num_classes=1000)

        assert len(labels) == 10000

        # Each class should appear exactly 10 times
        unique, counts = np.unique(labels, return_counts=True)
        assert len(unique) == 1000
        assert all(c == 10 for c in counts)

    def test_with_remainder(self) -> None:
        """Handles cases with remainder (not evenly divisible)."""
        labels = get_class_labels_for_balanced_generation(num_images=1005, num_classes=1000)

        assert len(labels) == 1005

        # Most classes should have 1, some should have 2
        unique, counts = np.unique(labels, return_counts=True)

        # Classes 0-4 should have 2, rest should have 1
        count_dict = dict(zip(unique, counts, strict=True))
        for cls in range(5):
            assert count_dict[cls] == 2
        for cls in range(5, 1000):
            assert count_dict[cls] == 1

    def test_fewer_images_than_classes(self) -> None:
        """Handles case where num_images < num_classes."""
        labels = get_class_labels_for_balanced_generation(num_images=500, num_classes=1000)

        assert len(labels) == 500

        # Labels should be in range [0, 499]
        unique = np.unique(labels)
        assert len(unique) == 500
        assert unique.min() == 0
        assert unique.max() == 499

    @pytest.mark.parametrize(
        "num_images,num_classes",
        [
            (1000, 1000),  # Exact division
            (1001, 1000),  # One extra
            (500, 1000),  # Fewer than classes
            (50000, 1000),  # Standard FID evaluation
            (320, 10),  # HP search (32 per class, 10 classes)
        ],
    )
    def test_various_configurations(self, num_images: int, num_classes: int) -> None:
        """Various configurations should produce correct length."""
        labels = get_class_labels_for_balanced_generation(num_images=num_images, num_classes=num_classes)

        assert len(labels) == num_images
        assert labels.min() >= 0
        assert labels.max() < num_classes


# =============================================================================
# save_single_image Tests
# =============================================================================


class TestSaveSingleImage:
    """Tests for single image save with class prefix."""

    def test_saves_with_correct_filename(self, tmp_path: Path) -> None:
        """Should save with img_class{label}_{idx}.png format."""
        img = torch.rand(3, 256, 256) * 2 - 1  # [-1, 1] range

        path = save_single_image(
            img=img,
            label=207,
            img_idx=5,
            output_dir=tmp_path,
            prefix="img",
        )

        assert Path(path).exists()
        assert Path(path).name == "img_class0207_0005.png"

    def test_custom_prefix(self, tmp_path: Path) -> None:
        """Should respect custom prefix."""
        img = torch.rand(3, 256, 256) * 2 - 1

        path = save_single_image(
            img=img,
            label=0,
            img_idx=0,
            output_dir=tmp_path,
            prefix="sample",
        )

        assert Path(path).name == "sample_class0000_0000.png"

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        """Should create output directory if it doesn't exist."""
        output_dir = tmp_path / "nested" / "dir"
        img = torch.rand(3, 256, 256) * 2 - 1

        path = save_single_image(
            img=img,
            label=100,
            img_idx=10,
            output_dir=output_dir,
            prefix="img",
        )

        assert output_dir.exists()
        assert Path(path).exists()


# =============================================================================
# count_images_per_class Tests
# =============================================================================


class TestCountImagesPerClass:
    """Tests for per-class image counting (HP search resume)."""

    def test_counts_per_class_correctly(self, tmp_path: Path) -> None:
        """Should correctly count images per class."""
        # Create images for multiple classes
        (tmp_path / "img_class0000_0000.png").touch()
        (tmp_path / "img_class0000_0001.png").touch()
        (tmp_path / "img_class0000_0002.png").touch()
        (tmp_path / "img_class0001_0000.png").touch()
        (tmp_path / "img_class0001_0001.png").touch()
        (tmp_path / "img_class0207_0000.png").touch()

        counts = count_images_per_class(tmp_path, prefix="img")

        assert counts[0] == 3
        assert counts[1] == 2
        assert counts[207] == 1
        assert 2 not in counts  # Class 2 has no images

    def test_handles_different_prefixes(self, tmp_path: Path) -> None:
        """Should only count files with matching prefix."""
        (tmp_path / "img_class0000_0000.png").touch()
        (tmp_path / "img_class0000_0001.png").touch()
        (tmp_path / "sample_class0000_0000.png").touch()
        (tmp_path / "sample_class0000_0001.png").touch()
        (tmp_path / "sample_class0000_0002.png").touch()

        img_counts = count_images_per_class(tmp_path, prefix="img")
        sample_counts = count_images_per_class(tmp_path, prefix="sample")

        assert img_counts[0] == 2
        assert sample_counts[0] == 3

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory should return empty dict."""
        counts = count_images_per_class(tmp_path, prefix="img")
        assert counts == {}

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Nonexistent directory should return empty dict."""
        nonexistent = tmp_path / "does_not_exist"
        counts = count_images_per_class(nonexistent, prefix="img")
        assert counts == {}

    def test_ignores_non_matching_files(self, tmp_path: Path) -> None:
        """Should ignore files that don't match the pattern."""
        (tmp_path / "img_class0000_0000.png").touch()
        (tmp_path / "random_file.png").touch()
        (tmp_path / "img_0000.png").touch()
        (tmp_path / "img_class_0000.png").touch()
        (tmp_path / "metadata.json").touch()

        counts = count_images_per_class(tmp_path, prefix="img")

        assert len(counts) == 1
        assert counts[0] == 1
