from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class FewShotImageDataset(Dataset):
    def __init__(self, rows: list[dict], data_root: Path, class_to_idx: dict[str, int], transform):
        self.rows = rows
        self.data_root = data_root
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        path = self.data_root / row["example_path"]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[row["example_label"]]
        return image, label


def load_single_image(path: Path, transform, device):
    image = Image.open(path).convert("RGB")
    if transform is not None:
        image = transform(image)
    return image.unsqueeze(0).to(device)

