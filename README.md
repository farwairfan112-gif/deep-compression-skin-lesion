# Deep Compression — Reproduction & Extension to Medical Imaging

PyTorch reimplementation of [Han, Mao & Dally (ICLR 2016)](https://arxiv.org/abs/1510.00149) — reproduced on MNIST, then extended to 7-class skin lesion classification on HAM10000 with clinical safety analysis.


---

## What We Did

Reproduced the full Deep Compression pipeline on MNIST (LeNet-300-100 and LeNet-5), then applied the same pipeline — with no changes to core principles — to a real-world clinical task: skin lesion classification on the HAM10000 dermoscopy dataset. The central question explored is whether compression disproportionately harms sensitivity for rare but clinically dangerous lesion types (melanoma, BCC, actinic keratosis).

---

## Pipeline

**Pruning → Quantization → Huffman Coding**

- **Stage 1 — Pruning:** Remove weights below a magnitude threshold per layer. Retrain with masks to prevent pruned connections from recovering.
- **Stage 2 — Quantization:** K-means clustering of remaining weights with linear centroid initialization. Fine-tune centroids via backpropagation (Eq. 3 of paper).
- **Stage 3 — Huffman Coding:** Lossless encoding of weight indices and sparse position differences. Achieves 20–30% additional reduction.

---

## Datasets

- **MNIST** — 60,000 handwritten digit images, 10 classes
- **HAM10000** — 10,015 dermoscopic skin lesion images, 7 classes, severely imbalanced (66.9% melanocytic nevi)

---

## Models

| Model | Parameters | Task |
|---|---|---|
| LeNet-300-100 | 266K | MNIST |
| LeNet-5 | 431K | MNIST |
| ResNet50 | 25.6M | HAM10000 |
| VGG16 | 138M | HAM10000 |

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/deep-compression.git
pip install -r requirements.txt
```

Notebooks were developed on Kaggle T4 GPU. For HAM10000, download the dataset [here](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and update the path in the config cell at the top of the notebook.

---

## References

- Han, S., Mao, H., & Dally, W. J. (2016). [Deep Compression](https://arxiv.org/abs/1510.00149). ICLR 2016 Best Paper Award.
- Tschandl, P. et al. (2018). [The HAM10000 Dataset](https://doi.org/10.1038/sdata.2018.161). Scientific Data.
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Simonyan, K. & Zisserman, A. (2015). Very Deep Convolutional Networks. ICLR.
