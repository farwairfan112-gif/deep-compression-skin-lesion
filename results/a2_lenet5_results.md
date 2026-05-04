# A2 Results — LeNet-5 on MNIST

## Stage-by-Stage Compression

| Stage | Size (KB) | Ratio | Accuracy |
|---|---|---|---|
| Original (32-bit float) | 1,683.9 | 1× | 99.25% |
| After Pruning + Retrain | 165.1 | 10.2× | 99.01% |
| After Quantization | 49.0 | 34.4× | 99.02% |
| After Huffman (Final) | 43.7 | **38.5×** | 99.02% |

## Comparison with Paper

| Metric | Paper | Ours |
|---|---|---|
| Reference Error | 0.80% | 0.75% ✅ better |
| Compressed Error | 0.74% | 0.98% |
| Original Size | 1,720 KB | 1,684 KB |
| Compressed Size | 44 KB | 43.7 KB |
| Compression Ratio | 39× | **38.5×** |

## Pruning Sparsity (Table 3)

| Layer | % Kept | % Pruned |
|---|---|---|
| conv1 | 66% | 34% |
| conv2 | 12% | 88% |
| ip1 (fc1) | 8% | 92% |
| ip2 (fc2) | 19% | 81% |

## Quantization (Table 3)

| Layer | Bits | Clusters |
|---|---|---|
| conv1, conv2 | 8-bit | 256 |
| ip1, ip2 | 5-bit | 32 |
