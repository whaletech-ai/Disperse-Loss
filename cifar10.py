import torch
import torchvision
# from pathlib import Path
# import ssl

# # --- 解决 SSL 证书验证失败的问题 (CERTIFICATE_VERIFY_FAILED)
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
# # -----------------------------------------------------------


from typing import Any

import numpy as np
import torch

from torchsmith.models.external._vae import VAE
from torchsmith.utils.pyutils import batched


class DatasetImagesWithVAE(torch.utils.data.Dataset):
    def __init__(
        self,
        data: np.ndarray,
        *,
        labels: np.ndarray,
        scale_factor: float,
        vae: VAE,
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.samples[idx], self.labels[idx]


# 定义路径
DATA_DIR = Path("./data")  # 你可以修改为你想要的路径
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("开始下载/加载训练集...")
train_dset = torchvision.datasets.CIFAR10(
    root=DATA_DIR / "cifar_dataset",
    transform=torchvision.transforms.ToTensor(),
    download=True,  # 如果本地没有，会自动下载
    train=True,
)

print("开始下载/加载测试集...")
test_dset = torchvision.datasets.CIFAR10(
    root=DATA_DIR / "cifar_dataset",
    transform=torchvision.transforms.ToTensor(),
    download=True,
    train=False,
)

print(f"下载完成! 训练集大小: {len(train_dset)}, 测试集大小: {len(test_dset)}")

train_images = train_dset.data / 255.0
train_labels = np.array(train_dset.targets, dtype=np.int32)
test_images = test_dset.data / 255.0
test_labels = np.array(test_dset.targets, dtype=np.int32)

train_data = train_images.transpose((0, 3, 1, 2))
test_data = test_images.transpose((0, 3, 1, 2))

mean, std = 0.5, 0.5
autoencoded_images = (
    vae.encode(
        (train_data[:1000] - mean) / std  # (B, C, H, W)
    )
    .cpu()
    .numpy()
)
scale_factor = float(np.std(autoencoded_images))  # 1.2963932
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

