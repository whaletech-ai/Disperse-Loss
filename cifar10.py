import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

from pretrained_VAE import load_pretrain_vqvae


def batched(array, batch_size):
    for i in range(0, len(array), batch_size):
        yield array[i : i + batch_size]


class DatasetImagesWithVAE(torch.utils.data.Dataset):
    def __init__(
        self,
        data: np.ndarray,
        *,
        labels: np.ndarray,
        scale_factor: float,
        vae,
        mean: float = 0.5,
        std: float = 0.5,
        batch_size: int = 1000,
    ) -> None:
        data = (data - mean) / std
        samples_list = []
        for batch in batched(data, batch_size):
            samples_list.append(vae.encode(np.array(batch)).cpu())
        self.samples = torch.cat(samples_list, dim=0) / scale_factor
        self.labels = torch.tensor(labels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx], self.labels[idx]


def ensure_cifar10_download(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    train_dset = torchvision.datasets.CIFAR10(
        root=data_dir,
        transform=torchvision.transforms.ToTensor(),
        download=True,
        train=True,
    )
    test_dset = torchvision.datasets.CIFAR10(
        root=data_dir,
        transform=torchvision.transforms.ToTensor(),
        download=True,
        train=False,
    )
    return train_dset, test_dset


def export_cifar10_images(dataset, out_dir: Path, overwrite: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.png"))
    if existing and not overwrite:
        print(f"Skip export, folder not empty: {out_dir}")
        return
    if existing and overwrite:
        for p in existing:
            p.unlink()
    for i, img in enumerate(dataset.data):
        Image.fromarray(img).save(out_dir / f"{i:06d}.png")
    print(f"Saved {len(dataset.data)} images to {out_dir}")


# -------------------------
# dataset + VAE latents for training
# -------------------------
DATA_DIR = Path("./data")
CIFAR_DIR = DATA_DIR / "cifar_dataset"

train_dset, test_dset = ensure_cifar10_download(CIFAR_DIR)

train_images = train_dset.data / 255.0
train_labels = np.array(train_dset.targets, dtype=np.int32)
test_images = test_dset.data / 255.0
test_labels = np.array(test_dset.targets, dtype=np.int32)

train_data = train_images.transpose((0, 3, 1, 2))
test_data = test_images.transpose((0, 3, 1, 2))

mean, std = 0.5, 0.5
vae = load_pretrain_vqvae()
autoencoded_images = (
    vae.encode((train_data[:1000] - mean) / std).cpu().numpy()
)
scale_factor = float(np.std(autoencoded_images))  # ~1.2963932
print(f"Using mean, std: {mean} {std} and scale factor: {scale_factor}")

train_dataset = DatasetImagesWithVAE(
    data=train_data,
    labels=train_labels,
    mean=mean,
    std=std,
    vae=vae,
    scale_factor=scale_factor,
)
test_dataset = DatasetImagesWithVAE(
    data=test_data,
    labels=test_labels,
    mean=mean,
    std=std,
    vae=vae,
    scale_factor=scale_factor,
)


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 downloader/exporter.")
    parser.add_argument(
        "--out-root",
        type=str,
        default="data/cifar10_images",
        help="Root folder to save images (train/test subfolders).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing exported images.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    export_cifar10_images(train_dset, out_root / "train", args.overwrite)
    export_cifar10_images(test_dset, out_root / "test", args.overwrite)


if __name__ == "__main__":
    main()
