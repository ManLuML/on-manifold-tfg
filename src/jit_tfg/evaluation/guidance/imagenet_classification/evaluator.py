"""ImageNet classification evaluator for guidance validity.

This module implements the guidance validity evaluation following the TFG paper:
- Uses DeiT-Small (facebook/deit-small-patch16-224) for evaluation
- Different from the guide model (ViT-B/16) to prevent over-confidence

Reference:
    TFG Paper Section D.3: "For validity, we use another pre-trained classifier
    other than the one used in providing guidance to avoid over-confidence."
"""

import re
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Model mappings: guide_model -> eval_model (from TFG paper)
GUIDE_TO_EVAL_MODEL = {
    "google/vit-base-patch16-224": "facebook/deit-small-patch16-224",
    "facebook/deit-small-patch16-224": "google/vit-base-patch16-224",
}

# Default evaluation model for ImageNet
DEFAULT_EVAL_MODEL = "facebook/deit-small-patch16-224"

# Cache directory for model checkpoints (gitignored)
CACHE_DIR = Path(__file__).parent / "checkpoints"


class ImageNetClassificationEvaluator:
    """Evaluator for ImageNet classification guidance validity.

    Computes classification accuracy on generated images using a pre-trained
    classifier that is DIFFERENT from the guide model (to avoid over-confidence).

    Attributes:
        model_name: HuggingFace model name for evaluation.
        device: Device for computation.
        classifier: Pre-trained classification model.
        processor: Image processor for the classifier.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EVAL_MODEL,
        device: str = "cuda",
        cache_dir: str | Path | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            model_name: HuggingFace model name for evaluation.
                Default: facebook/deit-small-patch16-224
            device: Device for computation.
            cache_dir: Directory to cache model checkpoints.
                Default: {module_dir}/checkpoints/
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load model and processor
        self._load_model()

    def _load_model(self) -> None:
        """Load the pre-trained classifier and processor."""
        print(f"Loading evaluation classifier: {self.model_name}")

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
        )
        self.classifier = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
        )

        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()

        # Freeze parameters
        for param in self.classifier.parameters():
            param.requires_grad = False

        # Build transform from processor config
        self.transform = transforms.Compose([
            transforms.Resize(
                (self.processor.size["height"], self.processor.size["width"]),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.Normalize(
                mean=self.processor.image_mean,
                std=self.processor.image_std,
            ),
        ])

    @torch.no_grad()
    def compute_validity(
        self,
        images: torch.Tensor,
        target_classes: list[int],
        batch_size: int = 64,
    ) -> dict:
        """Compute guidance validity (classification accuracy).

        Args:
            images: Generated images of shape (N, C, H, W) in [-1, 1] or [0, 1] range.
            target_classes: List of target class indices. Can be:
                - Single class applied to all images: [111]
                - Multiple classes (one per image): [111, 222, 333, ...]
            batch_size: Batch size for evaluation.

        Returns:
            Dictionary with validity metrics:
                - validity: Overall accuracy (float)
                - predictions: List of predicted class indices
                - correct: Number of correct predictions
                - total: Total number of images
        """
        self.classifier.eval()

        # Normalize images to [0, 1] if in [-1, 1]
        if images.min() < 0:
            images = (images + 1) / 2
        images = images.clamp(0, 1)

        # Expand target_classes if single class
        if len(target_classes) == 1:
            target_classes = target_classes * len(images)
        elif len(target_classes) != len(images):
            raise ValueError(
                f"target_classes length ({len(target_classes)}) must be 1 or match images length ({len(images)})"
            )

        all_preds = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size].to(self.device)

            # Apply transform
            batch = self.transform(batch)

            # Get predictions
            outputs = self.classifier(batch)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = probs.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)

        # Compute accuracy
        target_tensor = torch.tensor(target_classes)
        pred_tensor = torch.tensor(all_preds)

        correct = (pred_tensor == target_tensor).sum().item()
        total = len(images)
        validity = correct / total

        return {
            "validity": validity,
            "predictions": all_preds,
            "correct": correct,
            "total": total,
        }

    @torch.no_grad()
    def compute_validity_multi_target(
        self,
        images: torch.Tensor,
        target_classes: list[int],
        images_per_class: int,
        batch_size: int = 64,
    ) -> dict:
        """Compute validity for multiple target classes (TFG paper protocol).

        Following TFG paper: for ImageNet, evaluates on 4 classes (111, 222, 333, 444)
        with equal number of images per class.

        Args:
            images: Generated images of shape (N, C, H, W) in [-1, 1] or [0, 1] range.
            target_classes: List of target class indices, e.g., [111, 222, 333, 444].
            images_per_class: Number of images generated per class.
            batch_size: Batch size for evaluation.

        Returns:
            Dictionary with validity metrics:
                - validity: Overall accuracy (float)
                - validity_per_class: Dict mapping class -> accuracy
                - predictions: List of predicted class indices
                - correct: Number of correct predictions
                - total: Total number of images
        """
        # Build labels list: images_per_class images for each target class
        labels = []
        for target in target_classes:
            labels.extend([target] * images_per_class)

        # Truncate to actual image count
        labels = labels[: len(images)]

        result = self.compute_validity(images, labels, batch_size)

        # Compute per-class accuracy
        validity_per_class = {}
        for i, target in enumerate(target_classes):
            start_idx = i * images_per_class
            end_idx = min((i + 1) * images_per_class, len(images))

            if end_idx <= start_idx:
                continue

            class_preds = result["predictions"][start_idx:end_idx]
            class_correct = sum(1 for p in class_preds if p == target)
            validity_per_class[target] = class_correct / (end_idx - start_idx)

        result["validity_per_class"] = validity_per_class
        return result

    @torch.no_grad()
    def compute_validity_from_folder(
        self,
        folder_path: str | Path,
        target_classes: list[int],
        images_per_class: int,
        batch_size: int = 64,
        prefix: str = "img",
        show_progress: bool = True,
        num_workers: int = 4,
    ) -> dict:
        """Compute validity by loading images from folder in batches.

        Memory-efficient version that loads images batch by batch from disk,
        avoiding the need to load all images into memory at once.
        Uses DataLoader with multiple workers for parallel image loading.

        Expected filename format: {prefix}_class{XXXX}_{YYYY}.png
        e.g., img_class0111_0000.png

        Args:
            folder_path: Path to folder containing generated images.
            target_classes: List of target class indices.
            images_per_class: Number of images per class.
            batch_size: Batch size for evaluation.
            prefix: Filename prefix (default: "img").
            show_progress: Whether to show progress bar.
            num_workers: Number of workers for parallel data loading.

        Returns:
            Dictionary with validity metrics:
                - validity: Overall accuracy (float)
                - validity_per_class: Dict mapping class -> accuracy
                - predictions: List of predicted class indices
                - correct: Number of correct predictions
                - total: Total number of images
        """
        self.classifier.eval()
        folder_path = Path(folder_path)

        # Build list of (filepath, target_class) in correct order
        image_tasks: list[tuple[Path, int]] = []
        for cls in target_classes:
            for idx in range(images_per_class):
                img_path = folder_path / f"{prefix}_class{cls:04d}_{idx:04d}.png"
                if img_path.exists():
                    image_tasks.append((img_path, cls))

        if not image_tasks:
            return {
                "validity": 0.0,
                "validity_per_class": dict.fromkeys(target_classes, 0.0),
                "predictions": [],
                "correct": 0,
                "total": 0,
            }

        # Create dataset and dataloader for parallel loading
        dataset = _ValidityDataset(image_tasks)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        all_preds: list[int] = []
        all_targets: list[int] = []

        iterator = dataloader
        if show_progress:
            iterator = tqdm(dataloader, desc="Computing validity")

        for batch_images, batch_targets in iterator:
            # Move to device and apply transform
            batch_tensor = batch_images.to(self.device)
            batch_tensor = self.transform(batch_tensor)

            # Get predictions
            outputs = self.classifier(batch_tensor)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = probs.argmax(dim=-1).cpu().tolist()

            all_preds.extend(preds)
            all_targets.extend(batch_targets.tolist())

        # Compute overall accuracy
        target_tensor = torch.tensor(all_targets)
        pred_tensor = torch.tensor(all_preds)
        correct = (pred_tensor == target_tensor).sum().item()
        total = len(all_preds)
        validity = correct / total if total > 0 else 0.0

        # Compute per-class accuracy
        validity_per_class: dict[int, float] = {}
        class_correct: dict[int, int] = dict.fromkeys(target_classes, 0)
        class_total: dict[int, int] = dict.fromkeys(target_classes, 0)

        for pred, target in zip(all_preds, all_targets):
            class_total[target] += 1
            if pred == target:
                class_correct[target] += 1

        for cls in target_classes:
            if class_total[cls] > 0:
                validity_per_class[cls] = class_correct[cls] / class_total[cls]
            else:
                validity_per_class[cls] = 0.0

        return {
            "validity": validity,
            "validity_per_class": validity_per_class,
            "predictions": all_preds,
            "correct": correct,
            "total": total,
        }


class _ValidityDataset(Dataset):
    """Dataset for validity evaluation with parallel loading."""

    def __init__(self, image_tasks: list[tuple[Path, int]]) -> None:
        """Initialize dataset.

        Args:
            image_tasks: List of (image_path, target_class) tuples.
        """
        self.image_tasks = image_tasks
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_tasks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load image and return with target class.

        Returns:
            Tuple of (image_tensor [0,1], target_class).
        """
        img_path, target_cls = self.image_tasks[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.to_tensor(img)
        return img_tensor, target_cls

    @staticmethod
    def get_eval_model_for_guide(guide_model: str) -> str:
        """Get the evaluation model name for a given guide model.

        Following TFG paper's protocol of using different models for
        guidance and evaluation.

        Args:
            guide_model: HuggingFace model name used for guidance.

        Returns:
            HuggingFace model name for evaluation.
        """
        if guide_model in GUIDE_TO_EVAL_MODEL:
            return GUIDE_TO_EVAL_MODEL[guide_model]
        return DEFAULT_EVAL_MODEL
