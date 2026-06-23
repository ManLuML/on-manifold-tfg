"""Fine-grained classification evaluator for guidance validity.

Generic top-1 accuracy evaluator for fine-grained classification tasks.
Used by both the bird benchmark (525 species) and the Stanford Cars
benchmark (196 Make-Model-Year classes); concrete benchmark scripts
instantiate it with their domain-specific HF model name.

Reference:
    TFG Paper Appendix D.5: Fine-grained Bird Classification
    Guide Model: dennisjooo/Birds-Classifier-EfficientNetB2
    Eval Model: chriamue/bird-species-classifier
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

DEFAULT_GUIDE_MODEL = "dennisjooo/Birds-Classifier-EfficientNetB2"
DEFAULT_EVAL_MODEL = "chriamue/bird-species-classifier"

CACHE_DIR = Path(__file__).parent / "checkpoints"


class FinegrainedClassifierEvaluator:
    """Generic evaluator for fine-grained classification guidance validity.

    Computes classification accuracy on generated images using a pre-trained
    classifier that is DIFFERENT from the guide model. Domain-agnostic;
    works for any HF `AutoModelForImageClassification` whose processor exposes
    standard `size`, `image_mean`, `image_std` (covers ViT, ConvNeXt, EfficientNet,
    ResNet, etc.).

    Attributes:
        model_name: HuggingFace model name for evaluation.
        device: Device for computation.
        classifier: Pre-trained classification model.
        processor: Image processor for the classifier.
        num_classes: Number of fine-grained classes.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EVAL_MODEL,
        device: str = "cuda",
        cache_dir: str | Path | None = None,
        label_overrides: list[str] | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            model_name: HuggingFace model name for evaluation.
            device: Device for computation.
            cache_dir: Directory to cache model checkpoints.
            label_overrides: Optional list of human-readable labels (length
                == num_classes) to overwrite a HF model's `id2label` when
                the upstream config left it as `LABEL_0`, `LABEL_1`, ...
                placeholders. Index `i` in the list becomes the new
                `id2label[i]`.
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.label_overrides = label_overrides

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._load_model()

    def _load_model(self) -> None:
        """Load the pre-trained classifier and processor."""
        print(f"Loading fine-grained evaluation classifier: {self.model_name}")

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

        for param in self.classifier.parameters():
            param.requires_grad = False

        size = self.processor.size
        if "height" in size and "width" in size:
            h, w = size["height"], size["width"]
        elif "shortest_edge" in size:
            h = w = size["shortest_edge"]
        else:
            raise ValueError(f"Unrecognized processor.size shape: {size}")

        self.transform = transforms.Compose([
            transforms.Resize(
                (h, w),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.Normalize(
                mean=self.processor.image_mean,
                std=self.processor.image_std,
            ),
        ])

        self.num_classes = self.classifier.config.num_labels

        if self.label_overrides is not None:
            if len(self.label_overrides) != self.num_classes:
                raise ValueError(
                    f"label_overrides length ({len(self.label_overrides)}) "
                    f"does not match model num_labels ({self.num_classes})"
                )
            self.classifier.config.id2label = dict(enumerate(self.label_overrides))
            self.classifier.config.label2id = {n: i for i, n in enumerate(self.label_overrides)}

    @torch.no_grad()
    def compute_validity(
        self,
        images: torch.Tensor,
        target_classes: list[int],
        batch_size: int = 64,
    ) -> dict:
        """Compute guidance validity (top-1 classification accuracy).

        Args:
            images: Generated images of shape (N, C, H, W) in [-1, 1] or [0, 1].
            target_classes: List of target class indices (one per image).
            batch_size: Batch size for evaluation.

        Returns:
            Dictionary with validity metrics:
                - validity: Overall accuracy (float)
                - predictions: List of predicted class indices
                - correct: Number of correct predictions
                - total: Total number of images
        """
        self.classifier.eval()

        if images.min() < 0:
            images = (images + 1) / 2
        images = images.clamp(0, 1)

        if len(target_classes) != len(images):
            raise ValueError(f"target_classes length ({len(target_classes)}) must match images length ({len(images)})")

        all_preds = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size].to(self.device)
            batch = self.transform(batch)

            outputs = self.classifier(batch)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = probs.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)

        target_tensor = torch.tensor(target_classes)
        pred_tensor = torch.tensor(all_preds)

        correct = (pred_tensor == target_tensor).sum().item()
        total = len(images)
        validity = correct / total if total > 0 else 0.0

        return {
            "validity": validity,
            "predictions": all_preds,
            "correct": correct,
            "total": total,
        }

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

        Memory-efficient version that loads images batch by batch from disk.

        Expected filename format: {prefix}_class{XXXX}_{YYYY}.png
        e.g., img_class0111_0000.png

        Args:
            folder_path: Path to folder containing generated images.
            target_classes: List of target fine-grained class indices.
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

        dataset = _FinegrainedValidityDataset(image_tasks)
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
            iterator = tqdm(dataloader, desc="Computing fine-grained validity")

        for batch_images, batch_targets in iterator:
            batch_tensor = batch_images.to(self.device)
            batch_tensor = self.transform(batch_tensor)

            outputs = self.classifier(batch_tensor)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = probs.argmax(dim=-1).cpu().tolist()

            all_preds.extend(preds)
            all_targets.extend(batch_targets.tolist())

        target_tensor = torch.tensor(all_targets)
        pred_tensor = torch.tensor(all_preds)
        correct = (pred_tensor == target_tensor).sum().item()
        total = len(all_preds)
        validity = correct / total if total > 0 else 0.0

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

    def get_label_name(self, label_id: int) -> str:
        """Get the label name for a given label ID.

        Args:
            label_id: Label index.

        Returns:
            Human-readable label name.
        """
        if hasattr(self.classifier.config, "id2label"):
            return self.classifier.config.id2label.get(label_id, f"class_{label_id}")
        return f"class_{label_id}"


class FinegrainedBirdEvaluator(FinegrainedClassifierEvaluator):
    """Backward-compatible alias defaulting to the bird-species eval model."""

    def __init__(
        self,
        model_name: str = DEFAULT_EVAL_MODEL,
        device: str = "cuda",
        cache_dir: str | Path | None = None,
        label_overrides: list[str] | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            label_overrides=label_overrides,
        )


class _FinegrainedValidityDataset(Dataset):
    """Dataset for fine-grained validity evaluation with parallel loading."""

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
