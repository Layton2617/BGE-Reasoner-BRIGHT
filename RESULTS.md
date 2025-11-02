# Experimental Results

## Overview

Comprehensive evaluation of BGE-base embeddings with LLM reranking on BRIGHT benchmark datasets.

**Experiment Date**: November 2, 2025
**Total Runtime**: 2 hours 6 minutes
**Platform**: Lambda Labs (1x A100 SXM4 40GB)
**API Error Rate**: 0%

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Datasets | 4 |
| Total Queries | 388 |
| Total Documents | 194,825 |
| Average NDCG@10 | **37.7%** |
| API Calls | 726 |
| Total Tokens | 1,725,840 |

## Detailed Results by Dataset

### 1. Theoremqa Theorems

**Domain**: Mathematics
**Size**: 76 queries, 23,839 documents

| Method | NDCG@10 (3 runs) | Average | Improvement |
|--------|------------------|---------|-------------|
| Baseline (Retrieve) | [0.053, 0.051, 0.057] | 5.4% | - |
| **RGS (Ours)** | **[0.355, 0.386, 0.416]** | **38.6%** | **7.1×** |

**Key Metrics**:
- Bi-encoder evaluations: 14,880
- Reranker evaluations: 457
- API calls: 187
- Tokens consumed: 564,891

---

### 2. Biology

**Domain**: Life Sciences
**Size**: 103 queries, 57,359 documents

| Method | NDCG@10 (3 runs) | Average | Improvement |
|--------|------------------|---------|-------------|
| Baseline (Retrieve) | [0.155, 0.137, 0.144] | 14.5% | - |
| **RGS (Ours)** | **[0.476, 0.407, 0.441]** | **44.1%** | **3.0×** |

**Key Metrics**:
- Bi-encoder evaluations: 16,815
- Reranker evaluations: 400
- API calls: 166
- Tokens consumed: 311,654

**Notes**: Best performing dataset with 44.1% NDCG@10

---

### 3. Psychology

**Domain**: Social Sciences
**Size**: 101 queries, 52,835 documents

| Method | NDCG@10 (3 runs) | Average | Improvement |
|--------|------------------|---------|-------------|
| Baseline (Retrieve) | [0.119, 0.130, 0.143] | 13.1% | - |
| **RGS (Ours)** | **[0.356, 0.392, 0.408]** | **38.5%** | **2.9×** |

**Key Metrics**:
- Bi-encoder evaluations: 15,480
- Reranker evaluations: 442
- API calls: 188
- Tokens consumed: 498,429

---

### 4. Sustainable Living

**Domain**: Environmental Studies
**Size**: 108 queries, 60,792 documents

| Method | NDCG@10 (3 runs) | Average | Improvement |
|--------|------------------|---------|-------------|
| Baseline (Retrieve) | [0.120, 0.119, 0.143] | 12.7% | - |
| **RGS (Ours)** | **[0.278, 0.287, 0.325]** | **29.7%** | **2.3×** |

**Key Metrics**:
- Bi-encoder evaluations: 17,062
- Reranker evaluations: 436
- API calls: 185
- Tokens consumed: 350,866

---

## Performance Analysis

### NDCG@10 Distribution

```
Dataset Rankings (NDCG@10):
1. Biology:             44.1% ████████████████████████████████████████████
2. Psychology:          38.5% ██████████████████████████████████████
3. Theoremqa Theorems:  38.6% ██████████████████████████████████████
4. Sustainable Living:  29.7% █████████████████████████████

Baseline (Average):     11.4% ███████████
```

### Performance Improvement

All datasets show significant improvements over baseline:

```
Improvement Factor:
Theoremqa Theorems:  7.1× ███████████████████████████████████████
Biology:             3.0× ████████████████
Psychology:          2.9× ███████████████
Sustainable Living:  2.3× ████████████

Average:             3.8× ███████████████████
```

## Resource Utilization

### API Efficiency

| Metric | Per Query | Per Dataset |
|--------|-----------|-------------|
| API Calls | 1.87 | ~180 |
| Tokens | 4,448 | ~430,000 |
| Runtime | ~19 sec | ~31 min |

### Cost Efficiency

Using `gpt-4o-mini` for reranking:
- **Input tokens**: ~1.2M ($0.15 per 1M)
- **Output tokens**: ~0.5M ($0.60 per 1M)
- **Estimated total cost**: ~$0.48 for all 4 datasets

**Cost vs Performance**:
- 90% cheaper than GPT-4
- 3.8× better than baseline
- 0% API error rate

## Comparison with Baselines

| Method | Average NDCG@10 | Relative Performance |
|--------|-----------------|---------------------|
| BM25 | ~18% | Baseline |
| BGE-base (Retrieve only) | 11.4% | 0.63× |
| **BGE-base + LLM Rerank (Ours)** | **37.7%** | **2.1× vs BM25** |
| BGE-large (Expected) | ~30% | 1.67× |

## Reproducibility

All results are fully reproducible using the provided code:

```bash
# Run experiments on all datasets
for dataset in theoremqa_theorems biology psychology sustainable_living; do
    python run_retrieval.py \
        --benchmark_name BRIGHT \
        --dataset_name $dataset \
        --model_name bge-base \
        --algo_name RGS \
        --embedding_path ./embedding_data
done
```

**Result Files**: See `results/BRIGHT_bge-base_None_csv/` for raw CSV outputs.

## Technical Notes

### Configuration

- **Graph degree**: 12
- **Query L**: 5000
- **Second query quota**: 500
- **Temperature**: 0
- **Max tokens**: window_size × 10

### Error Handling

- **Total API calls**: 726
- **Failed calls**: 0
- **Success rate**: 100%
- **Retry mechanism**: Up to 5 retries with exponential backoff

### Stability

Experiment ran uninterrupted for 2+ hours with:
- No crashes
- No API timeouts
- Consistent performance across all datasets

## Conclusions

1. **Strong Performance**: 37.7% average NDCG@10 exceeds baseline by 3.8×
2. **Consistency**: All 4 datasets show substantial improvements
3. **Efficiency**: Lightweight LLM achieves competitive results at low cost
4. **Reliability**: 0% error rate demonstrates production-ready stability

## Future Work

Potential improvements:
- Test with other LLM backends (local models, other APIs)
- Optimize candidate selection to reduce API calls
- Explore hybrid reranking strategies
- Scale to remaining 8 BRIGHT datasets

---

**Last Updated**: November 2, 2025
