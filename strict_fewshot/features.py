from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def extract_image_stats(paths: list[Path]) -> torch.Tensor:
    feats = []
    for path in paths:
        img = Image.open(path).convert("RGB").resize((64, 64))
        x = torch.tensor(list(img.getdata()), dtype=torch.float32).view(-1, 3) / 255.0
        mean = x.mean(dim=0)
        std = x.std(dim=0)
        feats.append(torch.cat([mean, std], dim=0))
    return F.normalize(torch.stack(feats), p=2, dim=1)


class _ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return self.preprocess(image.convert("RGB"))


def resolve_remoteclip_checkpoint(cache_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"RemoteCLIP checkpoint not found: {checkpoint}")
        return checkpoint

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id="chendelong/RemoteCLIP",
        filename="RemoteCLIP-ViT-B-32.pt",
        cache_dir=str(cache_dir),
    ))


@torch.no_grad()
def extract_remoteclip(
    paths: list[Path],
    cache_dir: Path,
    checkpoint: Path | None,
    device,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[torch.Tensor, Path]:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32")

    checkpoint = resolve_remoteclip_checkpoint(cache_dir, checkpoint)

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "RemoteCLIP checkpoint is incompatible with open_clip ViT-B-32: "
            f"missing={incompatible.missing_keys[:5]}, "
            f"unexpected={incompatible.unexpected_keys[:5]}"
        )
    model = model.to(device).eval()

    dataset = _ImagePathDataset(paths, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    feats = []
    for images in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                feat = model.encode_image(images)
        else:
            feat = model.encode_image(images)
        feats.append(F.normalize(feat.float(), p=2, dim=1).cpu())
    return torch.cat(feats, dim=0), checkpoint
