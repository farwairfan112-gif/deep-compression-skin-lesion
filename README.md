# Deep Compression: Reproduction & Extension to Medical Imaging

## Overview

This repository covers a two-part study of [Han, Mao & Dally (ICLR 2016)](https://arxiv.org/abs/1510.00149) — **Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization, and Huffman Coding** (Best Paper Award).

| Focus | Models | Dataset |
|---|---|---|
| (#reproduction) | Reproduction of paper results | LeNet-300-100, LeNet-5 | MNIST |
| (#extension-to-medical-imaging) | Extension to real-world clinical task | ResNet50, VGG16 | HAM10000 |

---

## The Deep Compression Pipeline

Deep Compression compresses neural networks in three sequential stages with **35×–49× compression** and negligible accuracy loss:

```
Original Model
     │
     ▼
┌─────────────────┐
│  Stage 1:       │  Magnitude-based pruning → sparse weights (CSR format)
│  Pruning        │  Retrain with masks to restore accuracy
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 2:       │  K-means clustering of remaining weights
│  Quantization   │  Linear centroid init → fine-tune centroids via backprop (Eq. 3)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 3:       │  Huffman-code weight indices + sparse position differences
│  Huffman Coding │  Lossless: 20–30% additional storage reduction
└────────┬────────┘
         │
         ▼
  Compressed Model
```

---

## Reproduction

Full PyTorch reimplementation of the Deep Compression pipeline on MNIST, matching the paper's Tables 1, 2, and 3.

### Results

| Model | Original | Compressed | Ratio | Our Error | Paper Error |
|---|---|---|---|---|---|
| LeNet-300-100 | 1,041 KB | 26.2 KB | **39.7×** | 2.77% | 1.58% |
| LeNet-5 | 1,684 KB | 43.7 KB | **38.5×** | 0.98% | 0.74% |

> Compressed sizes within **3% of paper targets**. LeNet-5 baseline accuracy (99.25%) **exceeded** the paper's reported 99.20%.

### Stage-by-Stage: LeNet-5

| Stage | Size (KB) | Ratio | Accuracy |
|---|---|---|---|
| Original (32-bit float) | 1,683.9 | 1× | 99.25% |
| After Pruning + Retrain | 165.1 | 10.2× | 99.01% |
| After Quantization | 49.0 | 34.4× | 99.02% |
| **After Huffman (Final)** | **43.7** | **38.5×** | **99.02%** |

### Stage-by-Stage: LeNet-300-100

| Stage | Size (KB) | Ratio | Accuracy |
|---|---|---|---|
| Original (32-bit float) | 1,041.4 | 1× | 97.98% |
| After Pruning + Retrain | 98.4 | 10.6× | 97.27% |
| After Quantization | 30.0 | 34.7× | 97.23% |
| **After Huffman (Final)** | **26.2** | **39.7×** | **97.23%** |

### Pruning Configuration (from Paper Tables 2 & 3)

| Layer | Model | % Kept | % Pruned |
|---|---|---|---|
| ip1 (fc1) | LeNet-300-100 | 8% | 92% |
| ip2 (fc2) | LeNet-300-100 | 9% | 91% |
| ip3 (fc3) | LeNet-300-100 | 26% | 74% |
| conv1 | LeNet-5 | 66% | 34% |
| conv2 | LeNet-5 | 12% | 88% |
| ip1 (fc1) | LeNet-5 | 8% | 92% |
| ip2 (fc2) | LeNet-5 | 19% | 81% |

### Quantization Configuration

| Layer | Model | Bits | Clusters |
|---|---|---|---|
| ip1, ip2, ip3 | LeNet-300-100 | 6-bit | 64 |
| conv1, conv2 | LeNet-5 | 8-bit | 256 |
| ip1, ip2 | LeNet-5 | 5-bit | 32 |

---

## Extension to Medical Imaging

> **Research Question:** *Does Deep Compression introduce disproportionate sensitivity loss for rare but clinically dangerous lesion types (melanoma, BCC, actinic keratosis)?*

The same pipeline (no changes to core principles) is applied to **7-class skin lesion classification** on the [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) dermoscopy dataset — a severely class-imbalanced, real-world clinical task.

### Why HAM10000 is a harder and more important test

| Property | MNIST / ImageNet | HAM10000 |
|---|---|---|
| Class balance | Balanced | Severely imbalanced (66.9% one class) |
| Meaningful metric | Top-1 accuracy | **Balanced accuracy + per-class sensitivity** |
| Clinical stakes | None | Missed melanoma = fatal |
| Prior compression work | Extensively studied | **No prior Deep Compression work** |

### Dataset: HAM10000 Class Distribution

| Class | Full Name | Count | % | Risk |
|---|---|---|---|---|
| akiec | Actinic Keratosis | 327 | 3.3% | ⚠️ HIGH |
| bcc | Basal Cell Carcinoma | 514 | 5.1% | ⚠️ HIGH |
| bkl | Benign Keratosis | 1,099 | 11.0% | low |
| df | Dermatofibroma | 115 | 1.1% | low |
| mel | **Melanoma** | 1,113 | 11.1% | 🔴 **HIGH** |
| nv | Melanocytic Nevi | 6,705 | 66.9% | low |
| vasc | Vascular Lesion | 142 | 1.4% | low |

### Models

- **ResNet50** — 25.6M parameters, 89.7 MB. Fine-tuned from ImageNet weights, 2-phase training strategy. BatchNorm layers excluded from pruning/quantization.
- **VGG16** — 138M parameters, 512 MB. Same architecture family as the paper (Table 5). Single-phase differential LR training. All 13 conv + 3 FC layers prunable.

### Compression Results

#### ResNet50

| Stage | Size (KB) | Ratio | Top-1 Acc | Bal. Acc |
|---|---|---|---|---|
| Original (32-bit) | 91,884 | 1.0× | 49.93% | 69.31% |
| After Pruning + Retrain | 74,925 | 1.2× | 58.66% | 75.37% |
| After Quantization | 29,621 | 3.1× | 59.92% | 74.55% |
| **After Huffman (Final)** | **12,844** | **7.2×** | **59.92%** | **74.55%** |

#### VGG16

| Stage | Size (KB) | Ratio | Top-1 Acc | Bal. Acc |
|---|---|---|---|---|
| Original (32-bit) | 524,567 | 1.0× | 51.60% | 66.43% |
| After Pruning + Retrain | 97,855 | 5.4× | 55.46% | 67.93% |
| After Quantization | 30,016 | 17.5× | 54.93% | 67.12% |
| **After Huffman (Final)** | **19,968** | **26.3×** | **54.93%** | **67.12%** |

### 🔬 Core Clinical Finding: Compression Does NOT Harm Cancer Detection

Per-class sensitivity for **HIGH-RISK** classes (baseline → final compressed):

| Class | Risk | ResNet50 Δ | VGG16 Δ |
|---|---|---|---|
| **Melanoma** | 🔴 HIGH | 56.3% → **71.9% (+15.6 pp)** | 74.9% → **76.6% (+1.7 pp)** |
| **BCC** | ⚠️ HIGH | 85.9% → **87.3% (+1.4 pp)** | 83.1% → 80.3% (−2.8 pp) |
| **AKiec** | ⚠️ HIGH | 73.1% → **75.0% (+1.9 pp)** | 67.3% → **71.2% (+3.9 pp)** |

> **Key insight:** Magnitude pruning acts as a *regularizer* on class-imbalanced data. Weights responsible for the dominant "nevi" class are large and established; minority-class weights are smaller and more spread. Pruning these, followed by class-weighted retraining, reduces majority-class bias and *improves* sensitivity for rare dangerous lesions.

### Comparative Summary

| Metric | ResNet50 | VGG16 |
|---|---|---|
| Original → Final size | 89.7 MB → 12.8 MB | 512.3 MB → 19.9 MB |
| Compression ratio | **7.2×** | **26.3×** |
| Paper ratio (same arch, ImageNet) | 17× | 49× |
| Baseline balanced accuracy | 69.31% | 66.43% |
| Final balanced accuracy | **74.55% (+5.24%)** | **67.12% (+0.69%)** |
| Melanoma sensitivity change | **+15.6 pp** | +1.7 pp |

---

## Repository Structure

```
deep-compression/
│
├── notebooks/
│   ├── A2_Deep_Compression_Reproduction.ipynb   # LeNet-300-100 & LeNet-5 on MNIST
│   └── A3_Deep_Compression_Extension.ipynb      # ResNet50 & VGG16 on HAM10000
│
├── results/
│   ├── a2_lenet300_results.md                   # Stage-by-stage tables
│   ├── a2_lenet5_results.md
│   ├── a3_resnet50_results.md
│   └── a3_vgg16_results.md
│
├── assets/
│   └── images/                                  # Result plots, diagrams
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Usage

### Prerequisites

```bash
git clone https://github.com/YOUR_USERNAME/deep-compression.git
cd deep-compression
pip install -r requirements.txt
```

### Running Assignment 2 (MNIST Reproduction)

Open `notebooks/A2_Deep_Compression_Reproduction.ipynb` in Jupyter or upload to Kaggle.

The notebook will automatically download MNIST via `torchvision.datasets`.

### Running Assignment 3 (HAM10000 Extension)

1. Download the [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) or use the [Kaggle version](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)
2. Open `notebooks/A3_Deep_Compression_Extension.ipynb`
3. Update the dataset path in the config cell at the top
4. Run on a GPU (T4 or better recommended — VGG16 requires ~15 GB VRAM)

> **Recommended platform:** Kaggle Notebooks (free T4 GPU). The notebooks were developed and tested there.

---

## Requirements

See [`requirements.txt`](requirements.txt) for full dependencies. Core:

```
torch>=2.0.0
torchvision>=0.15.0
numpy
pandas
scikit-learn
matplotlib
seaborn
Pillow
tqdm
jupyter
```

---

## Implementation Highlights

### Key Classes (same core used in both A2 and A3)

| Class | Purpose |
|---|---|
| `PruningMask` | `kthvalue`-based magnitude pruning with per-layer thresholds |
| `KMeansQuantizer` | Linear centroid initialization, `scatter_add` Eq. 3 gradient updates |
| `HuffmanCoder` | Builds Huffman tree on weight index + position difference distributions |

### A3-Specific Adaptations

| Adaptation | Reason |
|---|---|
| BatchNorm exclusion (ResNet50) | BN layers are normalization statistics, not feature weights — zeroing them crashes accuracy |
| Chunked CPU k-means | VGG16 classifier layer has 102M weights; full distance matrix = 12.25 GB, over T4 limit |
| Huffman sampling (200K) | Large layers sampled for tree construction; code lengths scale correctly |
| `WeightedRandomSampler` | HAM10000 is 66.9% nevi; balances class exposure per batch |
| Weighted cross-entropy | Inverse-frequency class weights force focus on rare lesions |

---

## Hardware & Software

| Component | Details |
|---|---|
| GPU | NVIDIA Tesla T4 (15 GB VRAM) — Kaggle |
| CUDA | 12.8 |
| PyTorch | 2.10.0+cu128 |
| Python | 3.10 |
| Platform | Kaggle Notebooks |

---

## References

1. Han, S., Mao, H., & Dally, W. J. (2016). [Deep Compression](https://arxiv.org/abs/1510.00149). ICLR 2016 Best Paper.
2. Han, S., Pool, J., Tran, J., & Dally, W. J. (2015). Learning Both Weights and Connections. NeurIPS.
3. Tschandl, P. et al. (2018). [The HAM10000 Dataset](https://doi.org/10.1038/sdata.2018.161). Scientific Data.
4. He, K. et al. (2016). Deep Residual Learning. CVPR.
5. Simonyan, K. & Zisserman, A. (2015). Very Deep Convolutional Networks. ICLR.

---

*FAST-NUCES, Department of Artificial Intelligence & Data Science — 2025*
