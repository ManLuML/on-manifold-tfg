"""Inception-v3 feature extraction for FID and IS calculation.

This module provides utilities for extracting features from images
using a pretrained Inception-v3 model, following the standard
FID/IS evaluation protocol.

IMPORTANT: Uses torch-fidelity's Inception-v3 implementation to ensure
exact compatibility with pre-computed reference statistics.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class InceptionV3Features:
    """Inception-v3 wrapper using torch-fidelity for feature extraction.

    Uses torch-fidelity's Inception-v3 implementation to ensure exact
    compatibility with pre-computed reference statistics (mu, sigma).

    CRITICAL: torch-fidelity expects uint8 [0,255] images, NOT float [0,1].

    Attributes:
        FID_FEATURE_DIM: Dimension of FID features (2048).
        NUM_CLASSES: Number of ImageNet classes (1000).
    """

    FID_FEATURE_DIM = 2048
    NUM_CLASSES = 1000

    def __init__(self, device: str = "cuda") -> None:
        """Initialize using torch-fidelity's Inception-v3.

        Args:
            device: Device for computation.
        """
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

        self.device = device
        # Request both 2048 features (for FID) and logits (for IS)
        self.model = FeatureExtractorInceptionV3(
            name="inception-v3-compat",
            features_list=["2048", "logits_unbiased"],
        )
        self.model = self.model.to(device)
        self.model.eval()

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract features and logits from images.

        Args:
            x: Input images of shape (B, 3, H, W) as uint8 [0, 255].

        Returns:
            Tuple of (features, logits):
                - features: (B, 2048) from final avg pool
                - logits: (B, 1000) class logits for IS
        """
        with torch.no_grad():
            outputs = self.model(x)

        # torch-fidelity returns tuple in order of features_list
        features = outputs[0]  # 2048-dim features
        logits = outputs[1]  # 1000-dim logits

        return features, logits


class ImageFolderDataset(Dataset):
    """Dataset for loading images from a folder as uint8 tensors.

    IMPORTANT: Returns uint8 [0,255] tensors for torch-fidelity compatibility.

    Supports common image formats (PNG, JPG, JPEG, WEBP, BMP).
    """

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(
        self,
        folder: str | Path,
    ) -> None:
        """Initialize dataset.

        Args:
            folder: Path to folder containing images.
        """
        self.folder = Path(folder)

        self.image_paths = sorted([p for p in self.folder.iterdir() if p.suffix.lower() in self.SUPPORTED_EXTENSIONS])

        if not self.image_paths:
            raise ValueError(f"No images found in {folder}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load image as uint8 tensor (CHW format).

        Returns:
            Image tensor of shape (3, H, W) with dtype uint8 [0, 255].
        """
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        # Convert to CHW uint8 tensor for torch-fidelity
        img_array = np.array(img)  # HWC
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # CHW, uint8
        return img_tensor


class InceptionFeatureExtractor:
    """Extract Inception-v3 features from images using torch-fidelity.

    Uses torch-fidelity's Inception-v3 for exact compatibility with
    pre-computed reference statistics.

    IMPORTANT: This extractor expects uint8 [0,255] images!

    Attributes:
        device: Device for computation.
        batch_size: Batch size for feature extraction.
    """

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 64,
    ) -> None:
        """Initialize feature extractor.

        Args:
            device: Device for computation ('cuda', 'cpu', 'mps').
            batch_size: Batch size for processing images.
        """
        self.device = device
        self.batch_size = batch_size
        self.model = InceptionV3Features(device=device)

    @torch.no_grad()
    def extract_from_folder(
        self,
        folder: str | Path,
        show_progress: bool = True,
    ) -> dict[str, np.ndarray]:
        """Extract features from all images in a folder.

        Images are loaded as uint8 [0,255] for torch-fidelity compatibility.

        Args:
            folder: Path to folder containing images.
            show_progress: Whether to show progress bar.

        Returns:
            Dictionary with 'features' (N, 2048) and 'logits' (N, 1000).
        """
        dataset = ImageFolderDataset(folder)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        all_features = []
        all_logits = []

        iterator = tqdm(dataloader, desc="Extracting features") if show_progress else dataloader

        for batch in iterator:
            batch = batch.to(self.device)  # uint8 tensor
            features, logits = self.model(batch)
            all_features.append(features.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

        return {
            "features": np.concatenate(all_features, axis=0),
            "logits": np.concatenate(all_logits, axis=0),
        }

    @torch.no_grad()
    def extract_from_tensor(
        self,
        images: torch.Tensor,
        show_progress: bool = True,
    ) -> dict[str, np.ndarray]:
        """Extract features from a batch of image tensors.

        Args:
            images: Image tensor of shape (N, C, H, W) as uint8 [0, 255].
            show_progress: Whether to show progress bar.

        Returns:
            Dictionary with 'features' (N, 2048) and 'logits' (N, 1000).
        """
        all_features = []
        all_logits = []

        num_batches = (len(images) + self.batch_size - 1) // self.batch_size
        iterator = range(0, len(images), self.batch_size)

        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Extracting features")

        for i in iterator:
            batch = images[i : i + self.batch_size].to(self.device)
            features, logits = self.model(batch)
            all_features.append(features.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

        return {
            "features": np.concatenate(all_features, axis=0),
            "logits": np.concatenate(all_logits, axis=0),
        }
