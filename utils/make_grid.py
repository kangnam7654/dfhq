import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image


def make_grid(
    images,
    nrow=5,
    padding=2,
    normalize=False,
    value_range=None,
    scale_each=False,
    pad_value=0,
):
    """
    Create a grid from a list or batch of image tensors.

    Args:
        images (Tensor or list): (B, C, H, W) tensor or list of image tensors
        nrow (int): Number of images in each row
        padding (int): Padding between images
        normalize (bool): If True, normalize to [0, 1]
        value_range (tuple): Range for normalization
        scale_each (bool): If True, scale each image separately
        pad_value (float): Padding value

    Returns:
        Tensor: Grid image tensor of shape (C, H, W)
    """
    return vutils.make_grid(
        images,
        nrow=nrow,
        padding=padding,
        normalize=normalize,
        value_range=value_range,
        scale_each=scale_each,
        pad_value=pad_value,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Make a grid image from a folder of images."
    )
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Path to the image directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the grid image"
    )
    parser.add_argument("--nrow", type=int, default=5, help="Number of images per row")
    parser.add_argument("--padding", type=int, default=2, help="Padding between images")
    parser.add_argument(
        "--max_images",
        type=int,
        default=25,
        help="Maximum number of images in the grid",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=None,
        help="Resize output image to (W, H)",
    )
    parser.add_argument(
        "--value_range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[0, 255],
        help="Value range for normalization",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    images = list(image_dir.glob("*.png"))
    samples = random.sample(images, min(args.max_images, len(images)))
    samples = [
        torch.from_numpy(np.array(Image.open(str(img)).convert("RGB"))).permute(2, 0, 1)
        for img in samples
    ]
    grid = make_grid(
        samples,
        nrow=args.nrow,
        padding=args.padding,
        value_range=tuple(args.value_range),
    )
    grid_image = Image.fromarray(grid.permute(1, 2, 0).numpy().astype(np.uint8))
    if args.resize:
        grid_image = grid_image.resize(tuple(args.resize))
    grid_image.save(args.output)


if __name__ == "__main__":
    main()
