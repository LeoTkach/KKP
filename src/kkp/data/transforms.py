"""Image transforms for training and evaluation."""

from __future__ import annotations

from torchvision import transforms


def get_transforms(
    image_size: int,
    *,
    train: bool,
    strong_augment: bool = False,
) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    if train and strong_augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                normalize,
            ],
        )

    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ],
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ],
    )
