# Does Compression Kill Cancer Detection?
### Reproducing Deep Compression and Extending It to Skin Lesion Classification on HAM10000

> **Course Project — Deep Learning**  
> Farwa Irfan & Annash Ahmed | MS Data Science, FAST-NUCES Islamabad

---

## Overview

This project has **three tasks**:

| Task | Description |
|------|-------------|
| **A1 — Understanding** | Literature review and analysis of Han et al. (ICLR 2016) Deep Compression pipeline |
| **A2 — Reproduction** | Full PyTorch re-implementation of Deep Compression on MNIST (LeNet-300-100 & LeNet-5) |
| **A3 — Extension** | First application of Deep Compression to dermoscopy (HAM10000) with VGG16 & ResNet50, examining clinical safety on class-imbalanced data |

**Key finding:** Deep Compression does *not* harm cancer detection. Melanoma sensitivity improves by **+15.6 pp** (ResNet50) and **+1.7 pp** (VGG16) after full compression. Magnitude pruning acts as a **debiasing mechanism** that reduces majority-class bias.

---

## Results Summary

### A2 — Reproduction on MNIST (within 3% of paper)

| Model | Our Size | Paper Size | Our Ratio | Paper Ratio | Accuracy |
|-------|----------|------------|-----------|-------------|----------|
| LeNet-300-100 | 26.2 KB | 27 KB | 39.7× | 40× | 2.77% err |
| LeNet-5 | 43.7 KB | 44 KB | 38.5× | 39× | 0.98% err |

### A3 — Extension on HAM10000

| Model | Original Size | Final Size | Compression | Balanced Acc (Base→Final) |
|-------|--------------|------------|-------------|--------------------------|
| VGG16 | 512.3 MB | 19.5 MB | **26.3×** | 66.4% → 67.1% |
| ResNet50 | 89.7 MB | 12.6 MB | **7.2×** | 69.3% → 74.6% |

### Per-Class Sensitivity (HIGH-risk classes, Base → Compressed)

| Class | Risk | VGG16 Δ | ResNet50 Δ |
|-------|------|---------|-----------|
| Melanoma (mel) | 🔴 HIGH | +1.7 pp | **+15.6 pp** |
| Actinic Keratosis (akiec) | 🔴 HIGH | +3.9 pp | +1.9 pp |
| Basal Cell Carcinoma (bcc) | 🔴 HIGH | −2.8 pp (still >80%) | +1.4 pp |

---

## Repository Structure

```
project-root/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── config.yaml                      # Hyperparameters & settings
├── train.py                         # Fine-tune VGG16/ResNet50 on HAM10000
├── inference.py                     # Run inference + compression pipeline
├── data/
│   └── sample_data.csv              # Sample HAM10000 metadata (10 rows)
├── notebooks/
│   ├── 01_inference_demo.ipynb      # Demo: load model, run compression, evaluate
│   ├── DeepCompression_reproduction.ipynb   # A2: MNIST reproduction
│   ├── Deep_compression_VGG-extension.ipynb # A3: VGG16 on HAM10000
│   └── Deep_compression_ResNet-extension.ipynb # A3: ResNet50 on HAM10000
├── src/
│   ├── __init__.py
│   ├── dataset.py                   # HAM10000 dataset & dataloader
│   ├── model.py                     # Model definitions + compression pipeline
│   └── utils.py                     # Metrics, plotting, Huffman coding
├── results/
│   ├── baseline_metrics.json        # Pre-compression evaluation results
│   ├── improved_metrics.json        # Post-compression evaluation results
│   └── training_log.csv             # Epoch-level training history
└── checkpoints/                     # Saved model weights (not tracked in git)
    └── .gitkeep
```

---

## The Deep Compression Pipeline

```
Original Model
     │
     ▼  Stage 1: Magnitude Pruning
  Sparse Model  (conv: keep 66%, FC: keep 10%)
  + Retrain with masks
     │
     ▼  Stage 2: K-Means Quantization
  Quantized Model  (conv: 8-bit/256 clusters, FC: 5-bit/32 clusters)
  + Centroid fine-tuning
     │
     ▼  Stage 3: Huffman Coding
  Compressed Model  (indices + sparse diffs Huffman coded)
     │
     ▼
  Final compressed .bin file
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/deep-compression-ham10000.git
cd deep-compression-ham10000
pip install -r requirements.txt
```

### 2. Download HAM10000

Download from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and place as:
```
data/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

### 3. Train

```bash
# Fine-tune ResNet50 on HAM10000
python train.py --model resnet50 --epochs 25

# Fine-tune VGG16
python train.py --model vgg16 --epochs 25
```

### 4. Run compression pipeline + evaluate

```bash
# Full pipeline: prune → quantize → huffman → evaluate
python inference.py --model resnet50 --checkpoint checkpoints/resnet50_best.pth

# Or run the demo notebook
jupyter notebook notebooks/01_inference_demo.ipynb
```

---

## Notebooks

| Notebook | Task | Description |
|----------|------|-------------|
| `DeepCompression_reproduction.ipynb` | A2 | Full MNIST reproduction: LeNet-300-100 & LeNet-5 |
| `Deep_compression_VGG-extension.ipynb` | A3 | VGG16 on HAM10000 — 26.3× compression |
| `Deep_compression_ResNet-extension.ipynb` | A3 | ResNet50 on HAM10000 — 7.2× compression |
| `01_inference_demo.ipynb` | Demo | Step-by-step walkthrough of the pipeline |

---

## Key Design Decisions

- **Class imbalance:** Weighted cross-entropy + `WeightedRandomSampler` (both used throughout all training phases)
- **Metric:** Balanced accuracy (unweighted mean recall) — not top-1 accuracy
- **VGG16 fine-tuning:** Single-phase differential LR (features: 1e-3, classifier: 1e-2) to prevent FC overfitting
- **ResNet50 fine-tuning:** Two-phase (frozen backbone → full unfreeze with CosineAnnealingLR)
- **BatchNorm:** Excluded from pruning in ResNet50 (`is_prunable()` check)
- **Chunked CPU k-means:** VGG16 classifier.0 (102M weights) — GPU VRAM insufficient; chunked on CPU

---

## Hardware

All experiments run on **Kaggle** with NVIDIA Tesla T4 (15 GB VRAM), PyTorch 2.10.0+cu128, CUDA 12.8, Python 3.10.

---

## References

1. Han, Mao & Dally — [Deep Compression (ICLR 2016 Best Paper)](https://arxiv.org/abs/1510.00149)
2. Tschandl et al. — [HAM10000 Dataset](https://doi.org/10.1038/sdata.2018.161)
3. Simonyan & Zisserman — [VGG (ICLR 2015)](https://arxiv.org/abs/1409.1556)
4. He et al. — [ResNet (CVPR 2016)](https://arxiv.org/abs/1512.03385)
