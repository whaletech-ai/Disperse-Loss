import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def iter_image_paths(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    paths = []
    for root, _, files in os.walk(folder):
        for name in files:
            if Path(name).suffix.lower() in exts:
                paths.append(Path(root) / name)
    return sorted(paths)


def load_batches(paths, batch_size, device):
    to_tensor = transforms.ToTensor()  # float32 in [0, 1]
    batch = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        batch.append(to_tensor(img))
        if len(batch) == batch_size:
            yield torch.stack(batch, dim=0).to(device)
            batch = []
    if batch:
        yield torch.stack(batch, dim=0).to(device)


def fid_torchmetrics(path1, path2, batch_size, device):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception:
        return None

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    paths1 = iter_image_paths(path1)
    paths2 = iter_image_paths(path2)

    if not paths1 or not paths2:
        raise ValueError("Both folders must contain at least one image.")

    for batch in load_batches(paths1, batch_size, device):
        fid.update(batch, real=True)
    for batch in load_batches(paths2, batch_size, device):
        fid.update(batch, real=False)

    return float(fid.compute().item())


def fid_pytorch_fid(path1, path2, batch_size, device, num_workers):
    try:
        from pytorch_fid import fid_score
    except Exception:
        return None

    return float(
        fid_score.calculate_fid_given_paths(
            [path1, path2],
            batch_size=batch_size,
            device=device,
            dims=2048,
            num_workers=num_workers,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute FID between two folders of images."
    )
    parser.add_argument("path1", type=str, help="Folder of images A")
    parser.add_argument("path2", type=str, help="Folder of images B")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output file path (default: auto-named)",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isdir(args.path1) or not os.path.isdir(args.path2):
        raise SystemExit("Both inputs must be valid folders.")

    fid = fid_torchmetrics(args.path1, args.path2, args.batch_size, args.device)
    if fid is None:
        fid = fid_pytorch_fid(
            args.path1, args.path2, args.batch_size, args.device, args.num_workers
        )

    if fid is None:
        raise SystemExit(
            "No FID library found. Install one of: "
            "`pip install torchmetrics` or `pip install pytorch-fid`."
        )

    name1 = os.path.basename(os.path.normpath(args.path1))
    name2 = os.path.basename(os.path.normpath(args.path2))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.out or f"{name1}_{name2}_{fid:.6f}_{timestamp}.txt"
    line = f"{name1}\t{name2}\t{fid:.6f}\n"

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(line)

    print(f"FID: {fid:.6f}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
