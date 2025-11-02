# BGE-Reasoner BRIGHT Benchmark Evaluation

Evaluation of BGE embeddings with LLM reranking on BRIGHT benchmark datasets.

## Overview

This repository contains experimental code and results for evaluating BGE embeddings combined with LLM-based reranking on the BRIGHT benchmark. The experiments achieved **NDCG@10 = 37.7%** across 4 datasets, demonstrating significant improvements over baseline retrieval methods.

## Key Results

| Dataset | Baseline NDCG@10 | With LLM Reranking | Improvement |
|---------|------------------|---------------------|-------------|
| theoremqa_theorems | 5.4% | **38.6%** | 7.1× |
| biology | 14.5% | **44.1%** | 3.0× |
| psychology | 13.1% | **38.5%** | 2.9× |
| sustainable_living | 12.7% | **29.7%** | 2.3× |
| **Average** | 11.4% | **37.7%** | **3.8×** |

See [RESULTS.md](RESULTS.md) for detailed analysis.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### 1. Generate Embeddings

```bash
# Single dataset
python generate_embeddings.py --dataset biology --model bge-base

# All datasets
python generate_embeddings.py --dataset all --model bge-base
```

### 2. Run Retrieval Experiments

```bash
python run_retrieval.py \
    --benchmark_name BRIGHT \
    --dataset_name biology \
    --model_name bge-base \
    --algo_name RGS \
    --embedding_path ./embedding_data
```

### 3. Environment Variables

For LLM reranking, set your API key:

```bash
export OPENAI_API_KEY=your_key_here
```

## Project Structure

```
BGE-Reasoner-BRIGHT/
├── README.md                 # This file
├── RESULTS.md                # Detailed experimental results
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── config.yaml               # Dataset configurations
├── model_config.yaml         # Model configurations
├── generate_embeddings.py    # Embedding generation script
├── run_retrieval.py          # Main retrieval evaluation
├── run_retrieval_single.py   # Single query retrieval
├── embedding_data/           # Generated embeddings (not tracked)
└── results/                  # Experimental results
    └── BRIGHT_bge-base_None_csv/
        ├── biology_*.csv
        ├── psychology_*.csv
        ├── sustainable_living_*.csv
        └── theoremqa_theorems_*.csv
```

## Methodology

### Two-Stage Retrieval

1. **Dense Retrieval (Stage 1)**
   - Use BGE embeddings for fast candidate retrieval
   - DiskANN indexing for efficiency
   - High recall

2. **LLM Reranking (Stage 2)**
   - Rerank top candidates using LLM
   - Significantly improves precision
   - Average 3.8× NDCG@10 improvement

### Algorithm: RGS

The Retrieval-Guided Search (RGS) algorithm:
- Combines dense retrieval with LLM reranking
- Efficient API usage (~1.87 calls per query)
- Balances speed and accuracy

## Datasets

**BRIGHT Benchmark**: Multi-domain retrieval benchmark

| Dataset | Queries | Documents | Domain |
|---------|---------|-----------|--------|
| theoremqa_theorems | 76 | 23,839 | Mathematics |
| biology | 103 | 57,359 | Life Sciences |
| psychology | 101 | 52,835 | Social Sciences |
| sustainable_living | 108 | 60,792 | Environmental |

**Total**: 388 queries, 194,825 documents

## Technical Details

### Models Used

- **Embedding**: bge-base (110M parameters)
- **Reranking**: OpenAI gpt-4o-mini
- **Metric**: NDCG@10

### Hardware & Runtime

- **Platform**: Lambda Labs (1x A100 SXM4 40GB)
- **Total Runtime**: ~2 hours for 4 datasets
- **Embedding Generation**: ~30 min per dataset
- **Retrieval + Reranking**: ~31 min per dataset

### API Usage

- **Total API Calls**: 726
- **Total Tokens**: ~1.7M
- **Error Rate**: 0%
- **Estimated Cost**: ~$0.50 (using gpt-4o-mini)

## Requirements

### Dependencies

```
numpy==1.26.4
faiss-cpu==1.9.0
openai
sentence-transformers
PyYAML
tqdm
```

See `requirements.txt` for complete list.

### Hardware

- **For Embedding Generation**: GPU with 8GB+ VRAM (or CPU with patience)
- **For Retrieval**: 16GB+ RAM
- **For LLM Reranking**: API-based (no local GPU needed)

## Configuration

### Dataset Config (`config.yaml`)

Specifies dataset paths and parameters for BRIGHT benchmark evaluation.

### Model Config (`model_config.yaml`)

Defines embedding model settings and parameters.

## Reproducibility

To reproduce the results:

1. **Generate Embeddings**:
   ```bash
   python generate_embeddings.py --dataset all --model bge-base
   ```

2. **Run Experiments**:
   ```bash
   for dataset in theoremqa_theorems biology psychology sustainable_living; do
       python run_retrieval.py \
           --benchmark_name BRIGHT \
           --dataset_name $dataset \
           --model_name bge-base \
           --algo_name RGS \
           --embedding_path ./embedding_data
   done
   ```

3. **Check Results**: CSV files will be in `results/BRIGHT_bge-base_None_csv/`

## Results Analysis

### Performance Highlights

- **Average NDCG@10**: 37.7% (3.8× improvement over baseline)
- **Best Dataset**: Biology (44.1%)
- **Most Improved**: Theoremqa Theorems (7.1× improvement)
- **Consistency**: All datasets show significant gains

### Cost Efficiency

Using gpt-4o-mini instead of larger models:
- **90% cost reduction** vs GPT-4
- **Comparable performance** for ranking tasks
- **Fast inference** (~10s per API call)

## Citation

If you find this work useful, please cite:

```bibtex
@misc{zhu2025bge-bright,
  author = {Tianlei Zhu},
  title = {BGE-Reasoner Evaluation on BRIGHT Benchmark},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/BGE-Reasoner-BRIGHT}
}
```

## References

- [BRIGHT Benchmark](https://brightbenchmark.github.io/)
- [BGE Embeddings](https://github.com/FlagOpen/FlagEmbedding)
- [OpenAI API](https://platform.openai.com/docs/)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- BRIGHT Benchmark team
- FlagEmbedding (BGE) team
- OpenAI for API access
