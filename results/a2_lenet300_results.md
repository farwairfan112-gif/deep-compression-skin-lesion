# A2 Results — LeNet-300-100 on MNIST

## Stage-by-Stage Compression

| Stage | Size (KB) | Ratio | Accuracy |
|---|---|---|---|
| Original (32-bit float) | 1,041.4 | 1× | 97.98% |
| After Pruning + Retrain | 98.4 | 10.6× | 97.27% |
| After Quantization | 30.0 | 34.7× | 97.23% |
| After Huffman (Final) | 26.2 | **39.7×** | 97.23% |

## Comparison with Paper

| Metric | Paper | Ours |
|---|---|---|
| Reference Error | 1.64% | 2.02% |
| Compressed Error | 1.58% | 2.77% |
| Original Size | 1,070 KB | 1,041 KB |
| Compressed Size | 27 KB | 26.2 KB |
| Compression Ratio | 40× | **39.7×** |

## Pruning Sparsity (Table 2)

| Layer | % Kept | % Pruned |
|---|---|---|
| ip1 (fc1) | 8% | 92% |
| ip2 (fc2) | 9% | 91% |
| ip3 (fc3) | 26% | 74% |

## Quantization (Table 2)

| Layer | Bits | Clusters |
|---|---|---|
| ip1, ip2, ip3 | 6-bit | 64 |
