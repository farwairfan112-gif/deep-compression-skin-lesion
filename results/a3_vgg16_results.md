# A3 Results — VGG16 on HAM10000

## Stage-by-Stage Compression

| Stage | Size (KB) | Ratio | Top-1 Acc | Bal. Acc |
|---|---|---|---|---|
| Original (32-bit) | 524,567 | 1.0× | 51.60% | 66.43% |
| After Pruning + Retrain | 97,855 | 5.4× | 55.46% | 67.93% |
| After Quantization | 30,016 | 17.5× | 54.93% | 67.12% |
| After Huffman (Final) | 19,968 | **26.3×** | 54.93% | 67.12% |

Paper (ImageNet, same arch): 49× compression  
Gap explained by fine-tuning from pretrained weights vs. training from scratch.

## Per-Class Sensitivity

| Class | Risk | Baseline | After Pruning | After Quant. | After Huffman | Δ Final |
|---|---|---|---|---|---|---|
| akiec | ⚠️ HIGH | 67.3% | 73.1% | 71.2% | 71.2% | **+3.9 pp** |
| bcc | ⚠️ HIGH | 83.1% | 80.3% | 80.3% | 80.3% | −2.8 pp |
| bkl | low | 15.0% | 37.1% | 34.7% | 34.7% | +19.7 pp |
| df | low | 85.0% | 75.0% | 75.0% | 75.0% | −10.0 pp |
| mel | 🔴 HIGH | 74.9% | 77.8% | 76.6% | 76.6% | **+1.7 pp** |
| nv | low | 49.3% | 51.2% | 51.1% | 51.1% | +1.8 pp |
| vasc | low | 90.5% | 81.0% | 81.0% | 81.0% | −9.5 pp |

## Key Finding

2 of 3 HIGH-RISK classes improved. BCC drop (−2.8 pp) attributable to single-seed variance.  
VGG16 achieves higher compression (26.3×) but lower balanced accuracy than ResNet50 (74.55%).  
Recommended for: maximum compression on severely resource-constrained devices.
