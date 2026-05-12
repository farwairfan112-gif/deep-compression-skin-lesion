"""
model.py
--------
Model definitions and the Deep Compression pipeline:
  Stage 1 — Magnitude Pruning  (+ retraining)
  Stage 2 — K-Means Quantization  (+ centroid fine-tuning)
  Stage 3 — Huffman Coding  (size estimation)

Supports VGG16 and ResNet50, both adapted for 7-class HAM10000.

Key design decisions (from paper):
  - Conv layers : keep 66 % weights, 8-bit quantization (256 clusters)
  - FC layers   : keep 10 % weights, 5-bit quantization (32 clusters)
  - BatchNorm weights are NOT pruned (ResNet50 only)
  - Centroid update via scatter_add (Equation 3, Han et al.)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from collections import defaultdict
from sklearn.cluster import KMeans


# ── Model factory ──────────────────────────────────────────────────────────────
def get_model(name: str, num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """
    Load a torchvision model and replace the final layer for num_classes.

    Parameters
    ----------
    name        : "vgg16" or "resnet50"
    num_classes : number of output classes (7 for HAM10000)
    pretrained  : use ImageNet pretrained weights
    """
    weights = "IMAGENET1K_V1" if pretrained else None

    if name == "vgg16":
        model = tvm.vgg16(weights=weights)
        # Replace classifier[6] with 4096 → num_classes
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(model.classifier[6].weight)
        nn.init.zeros_(model.classifier[6].bias)

    elif name == "resnet50":
        model = tvm.resnet50(weights=weights)
        # Replace fc with 2048 → num_classes
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(model.fc.weight)
        nn.init.zeros_(model.fc.bias)

    else:
        raise ValueError(f"Unknown model: {name}. Choose 'vgg16' or 'resnet50'.")

    return model


# ── Layer inspection helpers ───────────────────────────────────────────────────
def is_prunable(module: nn.Module) -> bool:
    """Return True if this layer should be pruned (Conv2d or Linear, NOT BatchNorm)."""
    return isinstance(module, (nn.Conv2d, nn.Linear))


def is_fc(module: nn.Module) -> bool:
    """Return True if this is a fully-connected layer."""
    return isinstance(module, nn.Linear)


def named_prunable_layers(model: nn.Module):
    """Yield (name, module) for all prunable layers."""
    for name, module in model.named_modules():
        if is_prunable(module):
            yield name, module


# ── Stage 1: Magnitude Pruning ─────────────────────────────────────────────────
class PruningMask:
    """Stores per-layer binary masks and applies them after each gradient step."""

    def __init__(self):
        self.masks = {}  # name → BoolTensor

    def compute_masks(self, model: nn.Module,
                      conv_keep: float = 0.66,
                      fc_keep:   float = 0.10):
        """
        Compute magnitude-based masks.
        Weights with |w| below the (1-keep_ratio) percentile are zeroed.
        """
        for name, module in named_prunable_layers(model):
            weights = module.weight.data.abs()
            keep    = fc_keep if is_fc(module) else conv_keep
            # kth smallest = (1-keep)*100 percentile → threshold
            threshold = torch.kthvalue(
                weights.flatten(),
                max(1, int((1 - keep) * weights.numel()))
            ).values
            self.masks[name] = (weights >= threshold)

    def apply(self, model: nn.Module):
        """Zero out all masked weights (called after optimizer.step())."""
        for name, module in named_prunable_layers(model):
            if name in self.masks:
                module.weight.data *= self.masks[name].float()

    def sparsity(self) -> dict:
        """Return per-layer sparsity ratios."""
        out = {}
        for name, mask in self.masks.items():
            out[name] = 1.0 - mask.float().mean().item()
        return out


def prune_model(model:      nn.Module,
                conv_keep:  float = 0.66,
                fc_keep:    float = 0.10) -> PruningMask:
    """
    Apply magnitude pruning in-place and return the mask object.
    Call mask.apply(model) after every optimizer.step() during retraining.
    """
    mask = PruningMask()
    mask.compute_masks(model, conv_keep, fc_keep)
    mask.apply(model)
    print("[prune] Sparsity per layer:")
    for name, sp in mask.sparsity().items():
        print(f"  {name}: {sp:.1%} pruned")
    return mask


# ── Stage 2: K-Means Quantization ─────────────────────────────────────────────
class QuantizedLayer:
    """
    Stores k-means codebook for one layer.
    Weights are replaced by integer codes; centroids are the shared weights.
    """

    def __init__(self, codes: np.ndarray, centroids: np.ndarray, shape: tuple):
        self.codes     = codes          # int array, same shape as weights
        self.centroids = centroids      # float array, length = n_clusters
        self.shape     = shape


def _kmeans_layer(weights_flat: np.ndarray,
                  n_clusters:   int,
                  chunk_size:   int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    """
    K-means with linear centroid initialisation.
    For large layers (VGG16 classifier.0 = 102M weights),
    runs in CPU chunks to avoid OOM.
    """
    non_zero_idx  = np.where(weights_flat != 0)[0]
    non_zero_vals = weights_flat[non_zero_idx]

    if len(non_zero_vals) == 0:
        centroids = np.linspace(weights_flat.min(), weights_flat.max(), n_clusters)
        codes     = np.zeros(len(weights_flat), dtype=np.int32)
        return codes, centroids

    # Linear initialisation of centroids
    init_centroids = np.linspace(non_zero_vals.min(),
                                 non_zero_vals.max(),
                                 n_clusters).reshape(-1, 1)

    # Chunked k-means for very large layers
    if len(non_zero_vals) > chunk_size:
        idx      = np.random.choice(len(non_zero_vals), chunk_size, replace=False)
        sample   = non_zero_vals[idx].reshape(-1, 1)
        km       = KMeans(n_clusters=n_clusters, init=init_centroids,
                          n_init=1, max_iter=300)
        km.fit(sample)
        centroids = km.cluster_centers_.flatten()
        # Assign all non-zero weights to nearest centroid
        dists  = np.abs(non_zero_vals[:, None] - centroids[None, :])
        labels = dists.argmin(axis=1).astype(np.int32)
    else:
        km        = KMeans(n_clusters=n_clusters, init=init_centroids,
                           n_init=1, max_iter=300)
        km.fit(non_zero_vals.reshape(-1, 1))
        centroids = km.cluster_centers_.flatten()
        labels    = km.labels_.astype(np.int32)

    # Build full code array (0 for zero weights)
    codes = np.zeros(len(weights_flat), dtype=np.int32)
    codes[non_zero_idx] = labels

    return codes, centroids


def quantize_model(model:      nn.Module,
                   conv_bits:  int = 8,
                   fc_bits:    int = 5,
                   chunk_size: int = 50_000) -> dict[str, QuantizedLayer]:
    """
    Quantize all prunable layers with k-means.
    Replaces weights in-place with centroid values.

    Returns
    -------
    codebook : dict mapping layer_name → QuantizedLayer
    """
    codebook = {}

    for name, module in named_prunable_layers(model):
        bits      = fc_bits if is_fc(module) else conv_bits
        n_clusters = 2 ** bits

        w_np     = module.weight.data.cpu().numpy().flatten()
        codes, centroids = _kmeans_layer(w_np, n_clusters, chunk_size)

        # Replace weights with centroid values
        w_quantized = centroids[codes].reshape(module.weight.shape)
        module.weight.data = torch.tensor(
            w_quantized, dtype=module.weight.dtype
        ).to(module.weight.device)

        codebook[name] = QuantizedLayer(
            codes=codes, centroids=centroids, shape=module.weight.shape
        )
        print(f"[quant] {name}: {bits}-bit ({n_clusters} clusters)")

    return codebook


def update_centroids(model:    nn.Module,
                     codebook: dict[str, QuantizedLayer]):
    """
    Centroid update: gather gradients per cluster and add to centroids.
    Implements Equation 3 of Han et al. (scatter_add).
    Called after loss.backward() during quantization-aware fine-tuning.
    """
    for name, module in named_prunable_layers(model):
        if name not in codebook:
            continue
        if module.weight.grad is None:
            continue

        ql          = codebook[name]
        grad_flat   = module.weight.grad.data.cpu().numpy().flatten()
        codes_flat  = ql.codes.flatten()
        n_clusters  = len(ql.centroids)

        # Scatter-add: sum gradients per cluster
        grad_sum    = np.zeros(n_clusters)
        np.add.at(grad_sum, codes_flat, grad_flat)

        # Update centroids (SGD step; actual LR applied by caller)
        ql.centroids -= grad_sum  # caller scales by LR

        # Re-apply updated centroids to weights
        w_new = ql.centroids[codes_flat].reshape(ql.shape)
        module.weight.data = torch.tensor(
            w_new, dtype=module.weight.dtype
        ).to(module.weight.device)


# ── Stage 3: Huffman Coding (size estimation) ──────────────────────────────────
class _Node:
    """Huffman tree node."""
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq   = freq
        self.left   = left
        self.right  = right

    def __lt__(self, other):
        return self.freq < other.freq


def _build_huffman_tree(symbols: np.ndarray) -> _Node:
    """Build Huffman tree from a symbol array."""
    import heapq
    counts = defaultdict(int)
    for s in symbols:
        counts[int(s)] += 1

    heap = [_Node(symbol=k, freq=v) for k, v in counts.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        return heap[0]

    while len(heap) > 1:
        l = heapq.heappop(heap)
        r = heapq.heappop(heap)
        heapq.heappush(heap, _Node(freq=l.freq + r.freq, left=l, right=r))

    return heap[0]


def _huffman_code_lengths(node: _Node, depth: int = 0) -> dict:
    """Return symbol → code_length mapping."""
    if node is None:
        return {}
    if node.left is None and node.right is None:
        return {node.symbol: max(depth, 1)}
    out = {}
    out.update(_huffman_code_lengths(node.left,  depth + 1))
    out.update(_huffman_code_lengths(node.right, depth + 1))
    return out


def estimate_huffman_size(codebook:         dict[str, QuantizedLayer],
                          model:            nn.Module,
                          sample_threshold: int = 200_000,
                          sample_size:      int = 200_000) -> float:
    """
    Estimate the compressed model size in KB after Huffman coding.
    For large layers, builds the tree from a sample.

    Returns
    -------
    total_kb : estimated compressed size in kilobytes
    """
    total_bits = 0

    for name, ql in codebook.items():
        symbols = ql.codes.flatten()
        n       = len(symbols)

        if n > sample_threshold:
            idx     = np.random.choice(n, sample_size, replace=False)
            sample  = symbols[idx]
            tree    = _build_huffman_tree(sample)
            lengths = _huffman_code_lengths(tree)
            # Frequency-weighted average code length, scaled to full layer
            counts  = defaultdict(int)
            for s in sample:
                counts[int(s)] += 1
            avg_len = sum(lengths.get(s, 8) * c for s, c in counts.items()) / sample_size
            total_bits += avg_len * n
        else:
            tree    = _build_huffman_tree(symbols)
            lengths = _huffman_code_lengths(tree)
            counts  = defaultdict(int)
            for s in symbols:
                counts[int(s)] += 1
            total_bits += sum(lengths.get(s, 8) * c for s, c in counts.items())

    total_kb = total_bits / 8 / 1024
    return total_kb


# ── Full pipeline convenience wrapper ─────────────────────────────────────────
class DeepCompressionPipeline:
    """
    Convenience wrapper that tracks state across all three stages.

    Usage
    -----
    pipeline = DeepCompressionPipeline(model, config)
    pipeline.prune()
    # retrain here ...
    pipeline.quantize()
    # centroid fine-tune here ...
    size_kb = pipeline.huffman_size()
    """

    def __init__(self, model: nn.Module, cfg: dict):
        self.model    = model
        self.cfg      = cfg
        self.mask     = None
        self.codebook = None

    def prune(self):
        self.mask = prune_model(
            self.model,
            conv_keep=self.cfg["pruning"]["conv_keep_ratio"],
            fc_keep=self.cfg["pruning"]["fc_keep_ratio"],
        )

    def apply_mask(self):
        """Call after every optimizer.step() during retraining."""
        if self.mask:
            self.mask.apply(self.model)

    def quantize(self):
        self.codebook = quantize_model(
            self.model,
            conv_bits=self.cfg["quantization"]["conv_bits"],
            fc_bits=self.cfg["quantization"]["fc_bits"],
            chunk_size=self.cfg["quantization"]["cpu_kmeans_chunk_size"],
        )

    def huffman_size(self) -> float:
        if self.codebook is None:
            raise RuntimeError("Call quantize() first.")
        return estimate_huffman_size(
            self.codebook,
            self.model,
            sample_threshold=self.cfg["huffman"]["sample_threshold"],
            sample_size=self.cfg["huffman"]["sample_size"],
        )
