"""
train.py
--------
Fine-tune VGG16 or ResNet50 on HAM10000.

Usage
-----
  python train.py --model resnet50
  python train.py --model vgg16 --epochs 25

Training strategy (from paper reproduction):
  VGG16  : single-phase differential LR (features 1e-3, classifier 1e-2)
            to avoid FC overfitting on 103M-parameter block.
  ResNet50: two-phase — Phase 1 frozen backbone (Adam 1e-3, 5 ep),
            Phase 2 full unfreeze (SGD + CosineAnnealingLR, 20 ep).

Both use weighted cross-entropy + WeightedRandomSampler.
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

from src.dataset import get_dataloaders
from src.model   import get_model
from src.utils   import evaluate, model_size_kb, append_training_log, save_metrics


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune model on HAM10000")
    p.add_argument("--model",      type=str, default="resnet50",
                   choices=["vgg16", "resnet50"])
    p.add_argument("--epochs",     type=int, default=None,
                   help="Override total fine-tune epochs from config")
    p.add_argument("--config",     type=str, default="config.yaml")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Resume from checkpoint path")
    return p.parse_args()


# ── Training loop ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device,
                mask=None, scheduler=None, iter_scheduler=False):
    """One training epoch. Returns average loss and top-1 accuracy."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Enforce pruning masks if retraining a pruned model
        if mask is not None:
            mask.apply(model)

        # Per-iteration LR scheduler (Caffe InvLR)
        if scheduler is not None and iter_scheduler:
            scheduler.step()

        total_loss += loss.item()
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    avg_loss = total_loss / len(loader)
    acc      = correct / total
    return avg_loss, acc


# ── VGG16 fine-tuning ──────────────────────────────────────────────────────────
def train_vgg16(model, loaders, class_weights, cfg, device, log_path):
    vcfg   = cfg["training"]["vgg16"]
    epochs = vcfg["total_epochs"]

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Differential LR: features group and classifier group
    optimizer = Adam([
        {"params": model.features.parameters(),   "lr": vcfg["lr_features"]},
        {"params": model.classifier.parameters(), "lr": vcfg["lr_classifier"]},
    ])
    scheduler = MultiStepLR(optimizer,
                             milestones=vcfg["scheduler_milestones"],
                             gamma=vcfg["scheduler_gamma"])

    best_bal_acc = 0.0
    ckpt_path    = os.path.join(cfg["checkpoints"]["dir"], "vgg16_best.pth")
    os.makedirs(cfg["checkpoints"]["dir"], exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, loaders["train"], optimizer, criterion, device
        )
        scheduler.step()

        val_res  = evaluate(model, loaders["val"], device, label=f"VGG16 ep{epoch}")
        bal_acc  = val_res["balanced_accuracy"]

        row = {
            "epoch": epoch, "model": "vgg16",
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc, 4),
            "val_acc":    round(val_res["accuracy"], 4),
            "val_bal_acc": round(bal_acc, 4),
        }
        append_training_log(log_path, row)
        print(f"  Epoch {epoch}/{epochs} | loss={train_loss:.4f} | "
              f"train_acc={train_acc:.4f} | val_bal={bal_acc:.4f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best checkpoint (bal_acc={bal_acc:.4f})")

    print(f"[train] VGG16 fine-tuning done. Best val balanced acc: {best_bal_acc:.4f}")
    return ckpt_path


# ── ResNet50 fine-tuning ───────────────────────────────────────────────────────
def train_resnet50(model, loaders, class_weights, cfg, device, log_path):
    rcfg      = cfg["training"]["resnet50"]
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    ckpt_path = os.path.join(cfg["checkpoints"]["dir"], "resnet50_best.pth")
    os.makedirs(cfg["checkpoints"]["dir"], exist_ok=True)
    best_bal_acc = 0.0

    # ── Phase 1: frozen backbone ────────────────────────────────────────────
    print("[train] ResNet50 Phase 1: frozen backbone")
    for name, param in model.named_parameters():
        param.requires_grad = (name.startswith("fc"))

    optimizer_p1 = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=rcfg["lr_phase1"])

    for epoch in range(1, rcfg["phase1_epochs"] + 1):
        train_loss, train_acc = train_epoch(
            model, loaders["train"], optimizer_p1, criterion, device
        )
        val_res  = evaluate(model, loaders["val"], device,
                            label=f"ResNet50 P1 ep{epoch}")
        bal_acc  = val_res["balanced_accuracy"]
        row = {"epoch": f"P1-{epoch}", "model": "resnet50",
               "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
               "val_acc": round(val_res["accuracy"], 4), "val_bal_acc": round(bal_acc, 4)}
        append_training_log(log_path, row)
        print(f"  P1 Epoch {epoch}/{rcfg['phase1_epochs']} | "
              f"loss={train_loss:.4f} | val_bal={bal_acc:.4f}")

    # ── Phase 2: full unfreeze ──────────────────────────────────────────────
    print("[train] ResNet50 Phase 2: full unfreeze")
    for param in model.parameters():
        param.requires_grad = True

    optimizer_p2 = SGD(model.parameters(),
                       lr=rcfg["lr_phase2_start"],
                       momentum=rcfg["momentum"],
                       weight_decay=rcfg["weight_decay"])
    scheduler_p2 = CosineAnnealingLR(optimizer_p2,
                                      T_max=rcfg["phase2_epochs"],
                                      eta_min=rcfg["lr_phase2_end"])

    for epoch in range(1, rcfg["phase2_epochs"] + 1):
        train_loss, train_acc = train_epoch(
            model, loaders["train"], optimizer_p2, criterion, device
        )
        scheduler_p2.step()
        val_res  = evaluate(model, loaders["val"], device,
                            label=f"ResNet50 P2 ep{epoch}")
        bal_acc  = val_res["balanced_accuracy"]
        row = {"epoch": f"P2-{epoch}", "model": "resnet50",
               "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
               "val_acc": round(val_res["accuracy"], 4), "val_bal_acc": round(bal_acc, 4)}
        append_training_log(log_path, row)
        print(f"  P2 Epoch {epoch}/{rcfg['phase2_epochs']} | "
              f"loss={train_loss:.4f} | val_bal={bal_acc:.4f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best checkpoint (bal_acc={bal_acc:.4f})")

    print(f"[train] ResNet50 fine-tuning done. Best val balanced acc: {best_bal_acc:.4f}")
    return ckpt_path


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # Override epochs if passed via CLI
    if args.epochs is not None:
        if args.model == "vgg16":
            cfg["training"]["vgg16"]["total_epochs"] = args.epochs
        else:
            total = args.epochs
            cfg["training"]["resnet50"]["phase1_epochs"] = max(1, total // 5)
            cfg["training"]["resnet50"]["phase2_epochs"] = total - max(1, total // 5)

    # DataLoaders
    loaders, class_weights = get_dataloaders(
        metadata_csv = cfg["data"]["metadata_csv"],
        image_dirs   = cfg["data"]["image_dirs"],
        batch_size   = cfg["training"][args.model]["batch_size"],
        image_size   = cfg["data"]["image_size"],
        num_workers  = cfg["training"]["num_workers"],
        seed         = cfg["training"]["seed"],
        device       = device,
    )

    # Model
    model = get_model(args.model, num_classes=cfg["classes"]["num_classes"])
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"[train] Loaded checkpoint: {args.checkpoint}")
    model = model.to(device)

    print(f"[train] Model size: {model_size_kb(model)/1024:.1f} MB")

    # Baseline evaluation
    baseline_res = evaluate(model, loaders["test"], device, label="Baseline")
    save_metrics(baseline_res, cfg["results"]["baseline_metrics"])

    log_path = cfg["results"]["training_log"]

    # Fine-tune
    if args.model == "vgg16":
        ckpt = train_vgg16(model, loaders, class_weights, cfg, device, log_path)
    else:
        ckpt = train_resnet50(model, loaders, class_weights, cfg, device, log_path)

    # Final evaluation
    model.load_state_dict(torch.load(ckpt, map_location=device))
    final_res = evaluate(model, loaders["test"], device, label="Fine-tuned")
    save_metrics(final_res, cfg["results"]["improved_metrics"])

    print(f"\n[train] Done. Checkpoint saved to: {ckpt}")


if __name__ == "__main__":
    main()
