"""Train CXR-HierViT-Hybrid: Dual-Stream Multi-Scale Hierarchical Network

Specifically engineered for 4-class Chest X-ray classification:
1. Dual-Stream Backbone:
   - Stream A: RAD-DINO (ViT-B/14) with spatial attentive pooling & dispersion features
   - Stream B: ConvNeXt-Tiny for high-frequency pulmonary micro-textures
2. Multi-Task Hierarchical Loss:
   - Focal Loss on 4-class logits (tackles severe class imbalance)
   - Binary normal vs. diseased screener loss
   - Dedicated COVID-19 specialist focal loss
   - Bacteria vs. Virus hard-pair contrastive cross-entropy loss
3. Dynamic Clinical Gating Inference:
   - Boosts COVID-19 sensitivity for rare cases
   - Resolves bacteria vs. virus boundary confusion using specialist logits
"""

import argparse
import csv
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

CLASSES = ("normal", "bacteria", "virus", "COVID-19")
CSV_COLUMNS = ("new_filename", *CLASSES)
NORMAL_IDX = 0
BACTERIA_IDX = 1
VIRUS_IDX = 2
COVID_IDX = 3

REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
MODEL_ID = "microsoft/rad-dino"
PACKAGE_ROOT = Path(__file__).resolve().parent
OFFLINE_CACHE = PACKAGE_ROOT / "data" / "offline_cache"


def configure_cache():
    torch_cache = OFFLINE_CACHE / "torch"
    if (torch_cache / "hub/checkpoints/convnext_tiny-983f1562.pth").is_file():
        os.environ["TORCH_HOME"] = str(torch_cache)


def rad_dino_snapshot():
    return OFFLINE_CACHE / "huggingface" / "hub" / "models--microsoft--rad-dino" / "snapshots" / REVISION


def rad_dino_processor_source():
    snapshot = rad_dino_snapshot()
    if (snapshot / "preprocessor_config.json").is_file():
        return snapshot, {"local_files_only": True}
    return MODEL_ID, {"revision": REVISION}


def rad_dino_weights_source():
    snapshot = rad_dino_snapshot()
    if all((snapshot / name).is_file() for name in ("config.json", "model.safetensors")):
        return snapshot, {"local_files_only": True}
    return MODEL_ID, {"revision": REVISION, "use_safetensors": True}


@contextmanager
def dataset_path(extracted_dir):
    if extracted_dir is not None:
        if not extracted_dir.is_dir():
            raise FileNotFoundError(f"The extracted dataset directory does not exist: {extracted_dir}")
        yield extracted_dir
        return
    data = PACKAGE_ROOT / "data"
    archive_path = data / "nycu-aimi-2026-lab3-cxr.zip"
    sample_path = data / "test_data_sample.csv"
    if not archive_path.is_file() or not sample_path.is_file():
        raise FileNotFoundError("Pass --data-dir with the extracted course dataset and test_data_sample.csv.")
    with TemporaryDirectory(prefix="cxr-submit-data-") as temp:
        target = Path(temp)
        with ZipFile(archive_path) as archive:
            archive.extractall(target)
        shutil.copyfile(sample_path, target / "test_data_sample.csv")
        yield target


# =========================================================================
# 1. Dataset & Data Loading
# =========================================================================

class CXRDataset(Dataset):
    def __init__(self, root, split, dino_transform, cnn_transform, augment=False):
        self.image_dir = root / f"{split}_images"
        self.dino_transform = dino_transform
        self.cnn_transform = cnn_transform
        self.augment = augment
        self.labeled = split != "test"
        csv_name = f"{split}_data.csv" if self.labeled else "test_data_sample.csv"
        
        self.affine = transforms.RandomAffine(
            degrees=5, translate=(0.03, 0.03), interpolation=transforms.InterpolationMode.BILINEAR
        ) if augment else None

        self.filenames = []
        self.labels = []
        with (root / csv_name).open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                fn = row["new_filename"]
                if not fn or not (self.image_dir / fn).is_file():
                    continue
                self.filenames.append(fn)
                if self.labeled:
                    vals = [int(row[c]) for c in CLASSES]
                    self.labels.append(vals.index(1))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        fn = self.filenames[index]
        with Image.open(self.image_dir / fn) as img:
            img = img.convert("RGB")
            if self.affine:
                img = self.affine(img)
            dino_px = self.dino_transform(img)
            cnn_px = self.cnn_transform(img)
        return (dino_px, cnn_px), self.labels[index] if self.labeled else fn


# =========================================================================
# 2. CXR-HierViT-Hybrid Architecture
# =========================================================================

class SpatialAttentivePooling(nn.Module):
    """Pools ViT spatial patches via learned attention and dispersion."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, patch_tokens):
        # patch_tokens: [B, N, D]
        scores = self.attn(patch_tokens) # [B, N, 1]
        weights = F.softmax(scores, dim=1)
        pooled = (patch_tokens * weights).sum(dim=1) # [B, D]
        dispersion = patch_tokens.std(dim=1) # [B, D]
        return pooled, dispersion


class CXRHierViTHybrid(nn.Module):
    def __init__(self, vit_encoder, cnn_backbone, hidden_dim=768, num_classes=4):
        super().__init__()
        self.vit = vit_encoder
        self.cnn = cnn_backbone # convnext_tiny.features
        
        # Spatial attentive pooling for ViT patches
        self.spatial_pool = SpatialAttentivePooling(hidden_dim)
        
        # CNN pooling
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Fusion projection: CLS(768) + AttnPool(768) + Dispersion(768) + CNN(768*2) = 3840
        in_features = hidden_dim * 3 + 768 * 2
        self.fusion_fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 1. Main 4-Class Classification Head
        self.head_4class = nn.Linear(512, num_classes)
        
        # 2. Clinical Screener Head: Normal vs Diseased (Binary)
        self.head_screener = nn.Linear(512, 1)
        
        # 3. Rare Class Specialist: COVID-19 Detector (Binary)
        self.head_covid = nn.Linear(512, 1)
        
        # 4. Hard-Pair Specialist: Bacteria (0) vs Virus (1)
        self.head_bac_vir = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, dino_x, cnn_x):
        # Stream A: RAD-DINO
        vit_out = self.vit(pixel_values=dino_x).last_hidden_state
        cls_token = vit_out[:, 0] # [B, 768]
        patch_tokens = vit_out[:, 1:] # [B, 1369, 768]
        
        attn_patch, dispersion_patch = self.spatial_pool(patch_tokens)
        
        # Stream B: ConvNeXt
        cnn_feat = self.cnn(cnn_x) # [B, 768, H, W]
        cnn_avg = self.cnn_pool(cnn_feat).flatten(1) # [B, 768]
        cnn_max = self.cnn_max_pool(cnn_feat).flatten(1) # [B, 768]
        
        # Multi-Scale Fusion
        combined = torch.cat([cls_token, attn_patch, dispersion_patch, cnn_avg, cnn_max], dim=1)
        fused = self.fusion_fc(combined)
        
        # Multi-Task Outputs
        logits_4class = self.head_4class(fused)
        logit_screener = self.head_screener(fused)
        logit_covid = self.head_covid(fused)
        logits_bac_vir = self.head_bac_vir(fused)
        
        return logits_4class, logit_screener, logit_covid, logits_bac_vir

    def predict(self, dino_x, cnn_x, gating=True):
        """Dynamic gating inference that combines specialist signals."""
        logits_4class, logit_screener, logit_covid, logits_bac_vir = self.forward(dino_x, cnn_x)
        
        if not gating:
            return logits_4class
            
        probs_4class = F.softmax(logits_4class, dim=1)
        prob_screener = torch.sigmoid(logit_screener).squeeze(-1) # P(diseased)
        prob_covid = torch.sigmoid(logit_covid).squeeze(-1) # P(covid)
        probs_bac_vir = F.softmax(logits_bac_vir, dim=1) # [B, 2] (0: bacteria, 1: virus)
        
        adjusted_probs = probs_4class.clone()
        for i in range(len(adjusted_probs)):
            # 1. Normal screener refinement: if model is confident lung is normal
            if prob_screener[i] < 0.20:
                adjusted_probs[i, NORMAL_IDX] = max(adjusted_probs[i, NORMAL_IDX].item(), 0.85)
                adjusted_probs[i, 1:] *= 0.15
            elif prob_screener[i] > 0.80:
                adjusted_probs[i, NORMAL_IDX] *= 0.2
            
            # 2. COVID-19 rare class boost: if covid specialist has high confidence
            if prob_covid[i] > 0.40 and adjusted_probs[i, COVID_IDX] > 0.05:
                adjusted_probs[i, COVID_IDX] = max(adjusted_probs[i, COVID_IDX].item(), prob_covid[i].item())
            
            # 3. Bacteria vs. Virus hard boundary arbitration:
            # If top two predicted classes are bacteria & virus, interpolate with specialist
            top2 = torch.topk(adjusted_probs[i], 2).indices.tolist()
            if set(top2) == {BACTERIA_IDX, VIRUS_IDX}:
                bac_p = probs_bac_vir[i, 0].item()
                vir_p = probs_bac_vir[i, 1].item()
                # 60% main head + 40% specialist head
                adjusted_probs[i, BACTERIA_IDX] = 0.6 * adjusted_probs[i, BACTERIA_IDX].item() + 0.4 * bac_p
                adjusted_probs[i, VIRUS_IDX] = 0.6 * adjusted_probs[i, VIRUS_IDX].item() + 0.4 * vir_p

        return adjusted_probs


# =========================================================================
# 3. Multi-Task Hierarchical Loss
# =========================================================================

class MultiTaskHierarchicalLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights

    def focal_loss(self, logits, targets, weights=None, gamma=1.5):
        ce = F.cross_entropy(logits, targets, weight=weights, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** gamma) * ce).mean()

    def forward(self, logits_4class, logit_screener, logit_covid, logits_bac_vir, targets):
        # 1. Main 4-class Focal Loss
        loss_main = self.focal_loss(logits_4class, targets, weights=self.class_weights, gamma=1.5)
        
        # 2. Screener: Normal (0) vs. Diseased (1, 2, 3)
        diseased_target = (targets > 0).float()
        loss_screener = F.binary_cross_entropy_with_logits(logit_screener.squeeze(-1), diseased_target)
        
        # 3. COVID-19 specialist focal loss (Class 3)
        covid_target = (targets == COVID_IDX).float()
        p_covid = torch.sigmoid(logit_covid.squeeze(-1))
        p_t = torch.where(covid_target == 1, p_covid, 1 - p_covid).clamp(1e-6, 1 - 1e-6)
        alpha = torch.where(covid_target == 1, 0.75, 0.25)
        loss_covid = -(alpha * ((1 - p_t) ** 2.0) * torch.log(p_t)).mean()
        
        # 4. Bacteria (1) vs. Virus (2) Specialist
        bv_mask = (targets == BACTERIA_IDX) | (targets == VIRUS_IDX)
        if bv_mask.sum() > 0:
            bv_targets = (targets[bv_mask] == VIRUS_IDX).long() # 0: Bacteria, 1: Virus
            loss_bac_vir = F.cross_entropy(logits_bac_vir[bv_mask], bv_targets, label_smoothing=0.05)
        else:
            loss_bac_vir = torch.tensor(0.0, device=targets.device)
            
        return loss_main + 0.3 * loss_screener + 0.5 * loss_covid + 0.4 * loss_bac_vir


# =========================================================================
# 4. Evaluation & Metrics
# =========================================================================

@torch.no_grad()
def evaluate_model(model, loader, device, gating=True):
    model.eval()
    num_classes = len(CLASSES)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    
    for (dino_x, cnn_x), targets in loader:
        dino_x = dino_x.to(device, non_blocking=True)
        cnn_x = cnn_x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16):
            probs = model.predict(dino_x, cnn_x, gating=gating)
            preds = probs.argmax(dim=1).cpu()
            
        pairs = targets * num_classes + preds
        confusion += torch.bincount(pairs, minlength=num_classes ** 2).reshape_as(confusion)
        
    correct = confusion.diag().double()
    support = confusion.sum(dim=1)
    f1 = 2 * correct / (support + confusion.sum(dim=0)).clamp_min(1)
    acc = correct.sum().item() / confusion.sum().item()
    macro_f1 = f1.mean().item()
    per_class = dict(zip(CLASSES, f1.tolist()))
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class,
        "confusion_matrix": confusion.tolist(),
        "bacteria_to_virus": confusion[BACTERIA_IDX, VIRUS_IDX].item(),
        "virus_to_bacteria": confusion[VIRUS_IDX, BACTERIA_IDX].item(),
    }


# =========================================================================
# 5. Training Loop
# =========================================================================

def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Image Preprocessors
    from transformers import AutoImageProcessor, Dinov2Model
    print("Loading pretrained foundation backbones...")
    processor_source, processor_options = rad_dino_processor_source()
    dino_processor = AutoImageProcessor.from_pretrained(processor_source, use_fast=False, **processor_options)
    dino_transform = lambda img: dino_processor(images=img, return_tensors="pt").pixel_values[0]

    cnn_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Datasets
    print("Initializing datasets...")
    train_dataset = CXRDataset(args.data_dir, "train", dino_transform, cnn_transform, augment=True)
    val_dataset = CXRDataset(args.data_dir, "val", dino_transform, cnn_transform, augment=False)
    test_dataset = CXRDataset(args.data_dir, "test", dino_transform, cnn_transform, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # 3. Model Construction
    print("Building CXR-HierViT-Hybrid model...")
    weights_source, load_options = rad_dino_weights_source()
    vit_encoder = Dinov2Model.from_pretrained(weights_source, **load_options)
    
    # Freeze lower ViT layers (0..7), train top 4 layers (8..11)
    for param in vit_encoder.embeddings.parameters():
        param.requires_grad = False
    for i, layer in enumerate(vit_encoder.encoder.layer):
        if i < 8:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True

    # ConvNeXt-Tiny: freeze bottom stages, train stage 3
    convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    cnn_features = convnext.features
    for i, stage in enumerate(cnn_features):
        if i < 6:
            for param in stage.parameters():
                param.requires_grad = False
        else:
            for param in stage.parameters():
                param.requires_grad = True

    model = CXRHierViTHybrid(vit_encoder, cnn_features, hidden_dim=768).to(device)

    # Class Weights for Focal Loss
    counts = torch.bincount(torch.tensor(train_dataset.labels), minlength=4).float()
    class_weights = (counts.rsqrt() / counts.rsqrt().mean()).to(device)
    criterion = MultiTaskHierarchicalLoss(class_weights=class_weights)

    # Differential Learning Rates
    backbone_params = [p for p in vit_encoder.parameters() if p.requires_grad] + \
                      [p for p in cnn_features.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("vit") and not n.startswith("cnn")]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1.5e-5, "weight_decay": 1e-2},
        {"params": head_params, "lr": 1.2e-4, "weight_decay": 1e-3},
    ])

    total_steps = (len(train_loader) // args.accum_steps) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    print(f"\nTraining configuration:")
    print(f"  Epochs: {args.epochs} | Batch size: {args.batch_size} (effective: {args.batch_size * args.accum_steps})")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Training samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test: {len(test_dataset)}")
    print("=" * 70)

    best_macro_f1 = 0.0
    best_checkpoint_path = args.output_dir / "best_model.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        step_count = 0
        optimizer.zero_grad(set_to_none=True)
        start_t = perf_counter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for step, ((dino_x, cnn_x), targets) in enumerate(pbar):
            dino_x = dino_x.to(device, non_blocking=True)
            cnn_x = cnn_x.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device.type, dtype=torch.float16):
                logits_4class, logit_screener, logit_covid, logits_bac_vir = model(dino_x, cnn_x)
                loss = criterion(logits_4class, logit_screener, logit_covid, logits_bac_vir, targets)
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()
            train_loss += loss.item() * args.accum_steps

            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step_count += 1

            pbar.set_postfix({"loss": f"{train_loss / (step + 1):.4f}"})

        train_time = perf_counter() - start_t
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        val_metrics = evaluate_model(model, val_loader, device, gating=True)
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]
        per_class = val_metrics["per_class_f1"]

        print(f"\n📊 Epoch {epoch} Results ({train_time:.1f}s):")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")
        print(f"  Per-Class F1 -> Normal: {per_class['normal']:.4f} | Bacteria: {per_class['bacteria']:.4f} | "
              f"Virus: {per_class['virus']:.4f} | COVID-19: {per_class['COVID-19']:.4f}")
        print(f"  Error Counts -> Bacteria->Virus: {val_metrics['bacteria_to_virus']} | "
              f"Virus->Bacteria: {val_metrics['virus_to_bacteria']}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **val_metrics
        })

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            print(f"  🌟 New Best Model! Saving checkpoint to {best_checkpoint_path.name}")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_macro_f1": val_f1,
                "val_accuracy": val_acc,
                "per_class_f1": per_class,
                "history": history
            }, best_checkpoint_path)

    # Save metrics JSON
    with (args.output_dir / "metrics.json").open("w") as f:
        json.dump({"best_macro_f1": best_macro_f1, "history": history}, f, indent=2)

    # 4. Generate Final Kaggle Submission
    print("\n" + "=" * 70)
    print(f"Loading best checkpoint (Val Macro F1 = {best_macro_f1:.4f}) for test inference...")
    best_ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    sub_path = args.output_dir / "submission_cxr_hier_hybrid.csv"
    with sub_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_COLUMNS)
        for (dino_x, cnn_x), filenames in tqdm(test_loader, desc="Generating Kaggle Submission"):
            dino_x = dino_x.to(device, non_blocking=True)
            cnn_x = cnn_x.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16):
                probs = model.predict(dino_x, cnn_x, gating=True)
                preds = probs.argmax(dim=1).cpu().tolist()
            for fn, label in zip(filenames, preds):
                writer.writerow([fn, *(int(i == label) for i in range(len(CLASSES)))])

    print(f"\n🎉 Training & Submission Completed Successfully!")
    print(f"📁 Best Checkpoint: {best_checkpoint_path}")
    print(f"📄 Kaggle Submission File: {sub_path}")
    print(f"🏆 Best Validation Macro F1: {best_macro_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train the submitted CXR-HierViT-Hybrid V1 model.")
    parser.add_argument("--data-dir", type=Path, help="Use an extracted dataset instead of the bundled ZIP")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for training results")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required for training")
    if min(args.epochs, args.batch_size, args.accum_steps) < 1 or args.workers < 0:
        parser.error("epochs, batch size and accumulation must be positive; workers must be nonnegative")

    configure_cache()
    output = args.output_dir.resolve()
    with dataset_path(args.data_dir) as data, TemporaryDirectory(prefix="cxr-submit-train-") as temp:
        output.mkdir(parents=True, exist_ok=False)
        args.data_dir, args.output_dir = data, Path(temp)
        train(args)
        for old_name, new_name in (
            ("best_model.pt", "cxr_hybrid.pt"),
            ("submission_cxr_hier_hybrid.csv", "submission.csv"),
            ("metrics.json", "metrics.json"),
        ):
            shutil.copyfile(args.output_dir / old_name, output / new_name)
    print(f"Saved training outputs in {output}")


if __name__ == "__main__":
    main()
