# A3 Results — ResNet50 on HAM10000

## Stage-by-Stage Compression

| Stage | Size (KB) | Ratio | Top-1 Acc | Bal. Acc |
|---|---|---|---|---|
| Original (32-bit) | 91,884 | 1.0× | 49.93% | 69.31% |
| After Pruning + Retrain | 74,925 | 1.2× | 58.66% | 75.37% |
| After Quantization | 29,621 | 3.1× | 59.92% | 74.55% |
| After Huffman (Final) | 12,844 | **7.2×** | 59.92% | 74.55% |

Paper (ImageNet, same arch): 17× compression

## Per-Class Sensitivity

| Class | Risk | Baseline | After Pruning | After Quant. | After Huffman | Δ Final |
|---|---|---|---|---|---|---|
| akiec | ⚠️ HIGH | 73.1% | 82.7% | 75.0% | 75.0% | **+1.9 pp** |
| bcc | ⚠️ HIGH | 85.9% | 87.3% | 87.3% | 87.3% | **+1.4 pp** |
| bkl | low | 39.5% | 52.7% | 57.5% | 57.5% | +18.0 pp |
| df | low | 90.0% | 90.0% | 90.0% | 90.0% | 0.0 pp |
| mel | 🔴 HIGH | 56.3% | 65.9% | 71.9% | 71.9% | **+15.6 pp** |
| nv | low | 45.1% | 53.8% | 54.5% | 54.5% | +9.4 pp |
| vasc | low | 95.2% | 95.2% | 85.7% | 85.7% | −9.5 pp |

## Key Finding

All three HIGH-RISK classes **improved** sensitivity after compression.  
Melanoma sensitivity: 56.3% → **71.9% (+15.6 pp)**
