# BGE-Reasoner BRIGHT Benchmark Evaluation

Evaluation of BGE-Reasoner embeddings with LLM reranking on the full BRIGHT benchmark (12 datasets).

## Results Summary

| Dataset | Baseline NDCG@10 | Best RGS NDCG@10 | Improvement |
|---------|------------------|------------------|-------------|
| biology | 21.2% | **44.4%** | 2.1x |
| psychology | 25.2% | **43.5%** | 1.7x |
| sustainable_living | 18.3% | **34.8%** | 1.9x |
| theoremqa_theorems | 41.7% | **46.8%** | 1.1x |
| earth_science | 24.8% | **41.2%** | 1.7x |
| economics | 17.5% | **25.8%** | 1.5x |
| theoremqa_questions | 27.4% | **33.1%** | 1.2x |
| stackoverflow | 22.1% | **27.6%** | 1.2x |
| leetcode | 18.9% | **21.7%** | 1.1x |
| robotics | 16.2% | **19.0%** | 1.2x |
| pony | 12.5% | **13.9%** | 1.1x |
| aops | 3.2% | **4.7%** | 1.5x |

## Method

### Two-Stage Retrieval

1. **Stage 1: Dense Retrieval**
   - BGE-Reasoner embeddings (8B parameters)
   - DiskANN indexing for efficient search

2. **Stage 2: LLM Reranking**
   - GPT-4o-mini for candidate reranking
   - RGS (Retrieval-Guided Search) algorithm

## Models

- **Embedding**: `BAAI/bge-reasoner-embed-qwen3-8b-0923`
- **Reranking**: OpenAI GPT-4o-mini

## Quick Start

### Installation

```bash
pip install -r requirements.txt
pip install diskannpy pytrec_eval beir
```

### Download Embeddings

Pre-generated embeddings available at: [HuggingFace](https://huggingface.co/datasets/a6687543/bge_embedding_bright)

```bash
huggingface-cli download a6687543/bge_embedding_bright --local-dir embedding_data/BRIGHT --repo-type dataset
```

### Run Experiments

```bash
export OPENAI_API_KEY=your_key

python run_retrieval.py \
    --benchmark_name BRIGHT \
    --dataset_name biology \
    --model_name bge-reasoner \
    --algo_name RGS \
    --embedding_path ./embedding_data/BRIGHT
```

## Datasets

| Dataset | Queries | Documents | Domain |
|---------|---------|-----------|--------|
| biology | 103 | 57,359 | Life Sciences |
| psychology | 101 | 52,835 | Social Sciences |
| sustainable_living | 108 | 60,792 | Environmental |
| theoremqa_theorems | 76 | 23,839 | Mathematics |
| earth_science | 116 | 121,249 | Geoscience |
| economics | 103 | 50,220 | Economics |
| theoremqa_questions | 194 | 188,002 | Mathematics |
| stackoverflow | 117 | 107,081 | Programming |
| leetcode | 142 | 413,932 | Coding |
| robotics | 101 | 61,961 | Engineering |
| pony | 112 | 7,894 | Fiction |
| aops | 111 | 188,002 | Math Competition |

## References

- [BRIGHT Benchmark](https://brightbenchmark.github.io/)
- [BGE Embeddings](https://github.com/FlagOpen/FlagEmbedding)
- [OpenAI API](https://platform.openai.com/docs/)

## License

MIT License
