"""Prepare reference dataset for FID computation.

This script preprocesses ImageNet images for use as the reference set
when computing Fréchet Inception Distance (FID). It applies center
cropping and resizing to match the training data distribution.

Usage:
    python -m jit_tfg.jit.prepare_ref \\
        --data_path /path/to/imagenet \\
        --output_path imagenet-train-256 \\
        --img_size 256

The script processes the ImageNet training set and saves all images
as PNG files with consistent preprocessing, enabling reproducible
FID calculations.

Preprocessing Steps:
    1. Multi-scale resize (halve while >= 2x target size)
    2. Final resize to target size (bicubic interpolation)
    3. Center crop to exact target size
    4. Save as PNG (lossless compression)
"""

import argparse
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from jit_tfg.models.jit.utils.crop import center_crop_arr


def main() -> None:
    """Process and save ImageNet images for FID reference.

    Loads the ImageNet training set, applies center cropping and resizing,
    and saves each image as a PNG file with sequential naming.

    Command-line Arguments:
        --data_path: Path to ImageNet root directory (required).
            Should contain a 'train' subdirectory with class folders.
        --output_path: Folder where transformed images will be saved.
            Default: "imagenet-train-256".
        --img_size: Target resolution for center-crop and resize.
            Default: 256.

    Output:
        PNG files named "transformed_XXXXXXXX.png" in the output directory,
        where XXXXXXXX is a zero-padded sequential index.

    Example:
        Running with default settings on ImageNet:
        - Input: 1.28M images (1000 classes × ~1300 images each)
        - Output: 1.28M PNG files at 256×256 resolution
        - Each file is ~200KB (uncompressed PNG)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to ImageNet root directory")
    parser.add_argument(
        "--output_path", type=str, default="imagenet-train-256", help="Folder where transformed images will be saved"
    )
    parser.add_argument("--img_size", type=int, default=256, help="Resolution to center-crop and resize")
    args = parser.parse_args()

    # Define preprocessing transform
    # center_crop_arr handles multi-scale resize + center crop
    # CenterCrop ensures exact size (handles edge cases from rounding)
    # ToTensor converts to (C, H, W) float32 in [0, 1]
    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
    ])

    # Load ImageNet training set with transforms
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform_train)

    # Create dataloader with parallel loading
    # shuffle=False ensures deterministic ordering
    data_loader = DataLoader(dataset_train, batch_size=256, num_workers=32, shuffle=False, pin_memory=False)

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Convert tensor back to PIL for saving
    to_pil = transforms.ToPILImage()
    global_idx = 0

    from tqdm import tqdm

    # Process all batches
    for batch_images, _batch_labels in tqdm(data_loader):
        # batch_images: (B, C, H, W) float32 in [0, 1]
        for i in range(batch_images.size(0)):
            img_tensor = batch_images[i]  # (C, H, W)

            # Convert to PIL and save
            # ToPILImage handles [0, 1] -> [0, 255] conversion
            pil_img = to_pil(img_tensor)
            out_path = os.path.join(args.output_path, f"transformed_{global_idx:08d}.png")
            # compress_level=0 disables compression for faster writes
            pil_img.save(out_path, format="PNG", compress_level=0)
            global_idx += 1

        print(f"Saved batch up to index={global_idx} ...")

    print("Finished saving all images.")


if __name__ == "__main__":
    main()
