"""
inference.py
------------
Run the full Deep Compression pipeline on a trained model:
  Stage 1 — Prune → retrain
  Stage 2 — Quantize → centroid fine-tune
  Stage 3 — Huffman size estimation

Evaluates on the test set after every stage and prints/saves a
summary table including balanced accuracy and per-class sensitivity.

Usage
-----
  python inference.py --model resnet50 --checkpoint checkpoints/resnet50_best.pth
  python inference.py --model vgg16    --checkpoint checkpoints/vgg16_best.pth

If no checkpoint is provided, uses the ImageNet-pretrained model directly
(useful for a quick smoke-test without training).
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.optim import SGD

from src.dataset import get_dataloaders, CLASS_NAMES
from src.model   import (get_model, prune_model, quantize_model,
                          estimate_huffman_size, update_centroids)
from src.utils   import (evaluate, model_size_kb, save_metrics,
                          caffe_inv_lr_scheduler,
                          plot_per_class_sensitivity,
                          plot_compression_waterfall,
                          plot_sensitivity_heatmap,
                          plot_compression_accuracy_tradeoff)


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Deep Compression pipeline + evaluation")
    p.add_argument("--model",      type=str, default="resnet50",
                   choices=["vgg16", "resnet50"])
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--config",     type=str, default="config.yaml")
    p.add_argument("--no-plots",   action="store_true",
                   help="Skip matplotlib plots (useful on headless machines)")
    return p.parse_args()


# ── Retraining after pruning ───────────────────────────────────────────────────
def retrain(model, loader, class_weights, cfg, device, mask):
    """
    Retrain the pruned model with masks enforced after every gradient step.
    Uses Caffe InvLR policy (same as original paper).
    """
    pcfg      = cfg["pruning"]
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = SGD(model.parameters(),
                    lr=pcfg["retrain_lr"],
                    momentum=pcfg["momentum"],
                    weight_decay=pcfg["weight_decay"])
    scheduler = caffe_inv_lr_scheduler(optimizer,
                                        lr0=pcfg["retrain_lr"],
                                        gamma=pcfg["lr_gamma"],
                                        power=pcfg["lr_power"])

    model.train()
    iteration = 0
    for epoch in range(pcfg["retrain_epochs"]):
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            mask.apply(model)      # enforce sparsity after every step
            scheduler.step()
            iteration += 1
            total_loss += loss.item()
        print(f"  [retrain] epoch {epoch+1}/{pcfg['retrain_epochs']} "
              f"loss={total_loss/len(loader):.4f}")


# ── Centroid fine-tuning after quantization ────────────────────────────────────
def centroid_finetune(model, loader, class_weights, cfg, device, codebook):
    """
    Fine-tune codebook centroids (Equation 3, Han et al.).
    Gradients are collected per cluster via scatter_add.
    """
    qcfg      = cfg["quantization"]
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = SGD(model.parameters(),
                    lr=qcfg["qft_lr"],
                    momentum=qcfg["momentum"],
                    weight_decay=qcfg["weight_decay"])

    model.train()
    for epoch in range(qcfg["qft_epochs"]):
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            update_centroids(model, codebook)   # scatter_add centroid update
            optimizer.step()
            total_loss += loss.item()
        print(f"  [qft] epoch {epoch+1}/{qcfg['qft_epochs']} "
              f"loss={total_loss/len(loader):.4f}")


# ── Summary table ──────────────────────────────────────────────────────────────
def print_summary_table(stages: list[str],
                         sizes:  list[float],
                         results: list[dict]):
    header = f"{'Stage':<20} {'Size':>10} {'Ratio':>8} {'Acc':>8} {'Bal.Acc':>10}"
    sep    = "-" * len(header)
    print(f"\n{header}\n{sep}")
    base_size = sizes[0]
    for stage, size, res in zip(stages, sizes, results):
        ratio = base_size / size
        print(f"{stage:<20} {size/1024:>8.1f}MB {ratio:>7.1f}× "
              f"{res['accuracy']*100:>7.1f}% {res['balanced_accuracy']*100:>9.1f}%")
    print(sep)

    # Per-class sensitivity table
    print(f"\n{'Class':<10} {'Risk':<8}", end="")
    for stage in stages:
        print(f" {stage[:10]:>12}", end="")
    print()
    for cls in CLASS_NAMES:
        risk = "HIGH" if cls in {"akiec", "bcc", "mel"} else "low"
        print(f"{cls:<10} {risk:<8}", end="")
        for res in results:
            print(f" {res['per_class_sensitivity'][cls]:>11.1f}%", end="")
        delta = (results[-1]["per_class_sensitivity"][cls]
                 - results[0]["per_class_sensitivity"][cls])
        sign  = "+" if delta >= 0 else ""
        print(f"   Δ{sign}{delta:.1f}pp")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] Device: {device} | Model: {args.model}")

    # DataLoaders
    bs = cfg["training"][args.model]["batch_size"]
    loaders, class_weights = get_dataloaders(
        metadata_csv = cfg["data"]["metadata_csv"],
        image_dirs   = cfg["data"]["image_dirs"],
        batch_size   = bs,
        image_size   = cfg["data"]["image_size"],
        num_workers  = cfg["training"]["num_workers"],
        seed         = cfg["training"]["seed"],
        device       = device,
    )

    # Load model
    model = get_model(args.model, num_classes=cfg["classes"]["num_classes"])
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"[inference] Loaded: {args.checkpoint}")
    else:
        print("[inference] No checkpoint — using ImageNet pretrained weights only.")
    model = model.to(device)

    # Track results at each stage
    stages  = []
    sizes   = []
    results = []

    # ── Baseline ────────────────────────────────────────────────────────────
    print("\n[inference] ── Baseline ─────────────────────────────")
    base_size = model_size_kb(model)
    base_res  = evaluate(model, loaders["test"], device, label="Baseline")
    stages.append("Baseline");  sizes.append(base_size);  results.append(base_res)

    # ── Stage 1: Pruning ─────────────────────────────────────────────────────
    print("\n[inference] ── Stage 1: Pruning ────────────────────")
    mask = prune_model(model,
                       conv_keep=cfg["pruning"]["conv_keep_ratio"],
                       fc_keep=cfg["pruning"]["fc_keep_ratio"])
    print("[inference] Retraining pruned model ...")
    retrain(model, loaders["train"], class_weights, cfg, device, mask)
    prune_size = model_size_kb(model)   # actual sparse representation
    prune_res  = evaluate(model, loaders["test"], device, label="After Pruning")
    stages.append("After Pruning");  sizes.append(prune_size);  results.append(prune_res)

    # Save pruned checkpoint
    ckpt_dir = cfg["checkpoints"]["dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{args.model}_pruned.pth"))

    # ── Stage 2: Quantization ────────────────────────────────────────────────
    print("\n[inference] ── Stage 2: Quantization ───────────────")
    codebook = quantize_model(
        model,
        conv_bits  = cfg["quantization"]["conv_bits"],
        fc_bits    = cfg["quantization"]["fc_bits"],
        chunk_size = cfg["quantization"]["cpu_kmeans_chunk_size"],
    )
    print("[inference] Centroid fine-tuning ...")
    centroid_finetune(model, loaders["train"], class_weights, cfg, device, codebook)
    quant_size = model_size_kb(model)
    quant_res  = evaluate(model, loaders["test"], device, label="After Quantization")
    stages.append("After Quant.");  sizes.append(quant_size);  results.append(quant_res)

    # ── Stage 3: Huffman Coding ──────────────────────────────────────────────
    print("\n[inference] ── Stage 3: Huffman Coding ─────────────")
    huffman_size_kb = estimate_huffman_size(
        codebook, model,
        sample_threshold = cfg["huffman"]["sample_threshold"],
        sample_size      = cfg["huffman"]["sample_size"],
    )
    # Huffman doesn't change weights — accuracy is same as post-quantization
    huffman_res = quant_res.copy()
    stages.append("After Huffman");  sizes.append(huffman_size_kb);  results.append(huffman_res)

    # ── Summary ──────────────────────────────────────────────────────────────
    print_summary_table(stages, sizes, results)

    # Save final metrics
    summary = {
        "stages":   stages,
        "sizes_kb": sizes,
        "compression_ratios": [sizes[0] / s for s in sizes],
        "results":  results,
    }
    save_metrics(summary, os.path.join(cfg["results"]["dir"],
                                        f"{args.model}_compression_summary.json"))

    # ── Plots ────────────────────────────────────────────────────────────────
    if not args.no_plots:
        rdir = cfg["results"]["dir"]
        os.makedirs(rdir, exist_ok=True)

        plot_compression_waterfall(
            sizes, model_name=args.model.upper(),
            save_path=os.path.join(rdir, f"{args.model}_waterfall.png"))

        plot_per_class_sensitivity(
            results, model_name=args.model.upper(),
            save_path=os.path.join(rdir, f"{args.model}_sensitivity.png"))

        plot_sensitivity_heatmap(
            results[0], results[-1], model_name=args.model.upper(),
            save_path=os.path.join(rdir, f"{args.model}_heatmap.png"))

        mel_idx = list({"akiec": 0, "bcc": 1, "bkl": 2,
                         "df": 3, "mel": 4, "nv": 5, "vasc": 6}).get
        mel_sens = [r["per_class_sensitivity"]["mel"] for r in results]
        bal_accs = [r["balanced_accuracy"] for r in results]
        plot_compression_accuracy_tradeoff(
            sizes, bal_accs, mel_sens, model_name=args.model.upper(),
            save_path=os.path.join(rdir, f"{args.model}_tradeoff.png"))

    print("\n[inference] Pipeline complete.")


if __name__ == "__main__":
    main()
