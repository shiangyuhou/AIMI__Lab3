"""Generate the competition CSV from either submitted CXR-HierViT-Hybrid checkpoint."""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from train_cxr_hybrid import (
    CSV_COLUMNS, PACKAGE_ROOT, RAD_DINO_CONFIG, CXRDataset, CXRHierViTHybrid, dataset_path,
)


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state"]
    expected = model.state_dict().keys()
    projection_names = {
        "q_proj": "attention.query", "k_proj": "attention.key",
        "v_proj": "attention.value", "o_proj": "output.dense",
    }
    for key in list(state):
        if key.startswith("vit.encoder.layer."):
            base, separator, tail = key.partition(".attention.")
            projection, _, parameter = tail.partition(".")
            if separator and projection in projection_names:
                compatible = f"{base}.attention.{projection_names[projection]}.{parameter}"
                if compatible in expected and compatible not in state:
                    state[compatible] = state.pop(key)
    model.load_state_dict(state, strict=True)


@torch.inference_mode()
def export(model, loader, path, device):
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_COLUMNS)
        for (vit, cnn), names in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                scores = model.predict(vit.to(device, non_blocking=True), cnn.to(device, non_blocking=True), gating=True)
            labels = scores.argmax(dim=1).cpu().tolist()
            writer.writerows([name, *(int(i == label) for i in range(4))]
                             for name, label in zip(names, labels, strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PACKAGE_ROOT / "cxr_hybrid_best.pt")
    parser.add_argument("--data-dir", type=Path, help="Use an extracted dataset instead of the bundled ZIP")
    parser.add_argument("--output", type=Path, required=True, help="Path for the new CSV")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required for FP16 inference")
    if args.output.exists():
        parser.error("The output CSV already exists. Choose a new path.")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}. Pass --checkpoint with your local .pt file.")

    from transformers import AutoImageProcessor, Dinov2Config, Dinov2Model

    config = Dinov2Config.from_pretrained(RAD_DINO_CONFIG, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(RAD_DINO_CONFIG, use_fast=False, local_files_only=True)
    model = CXRHierViTHybrid(Dinov2Model(config), models.convnext_tiny(weights=None).features)
    load_checkpoint(model, args.checkpoint)
    device = torch.device("cuda")
    model.to(device).eval()

    def dino_transform(img):
        return processor(images=img, return_tensors="pt").pixel_values[0]

    cnn_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    with dataset_path(args.data_dir) as data:
        test = CXRDataset(data, "test", dino_transform, cnn_transform)
        with (data / "test_data_sample.csv").open(encoding="utf-8-sig", newline="") as file:
            expected = [row[CSV_COLUMNS[0]] for row in csv.DictReader(file)]
        if test.filenames != expected:
            raise ValueError("Some test images listed in the submission template are missing")
        loader = DataLoader(test, batch_size=12, shuffle=False, num_workers=4, pin_memory=True)
        export(model, loader, args.output, device)
        print(f"Saved {len(test)} predictions to {args.output}")


if __name__ == "__main__":
    main()
