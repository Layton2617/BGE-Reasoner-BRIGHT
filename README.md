# BGE-Reasoner BRIGHT Reproduction

Reproduce **NDCG@10 ≈ 37.1%** on BRIGHT benchmark using [BGE-Reasoner](https://huggingface.co/BAAI/bge-reasoner-embed-qwen3-8b-0923).

## Quick Start

```bash
pip install -r requirements.txt
huggingface-cli login

# Single dataset
python generate_embeddings.py --dataset psychology --model bge-reasoner

# All datasets
python generate_embeddings.py --dataset all --model bge-reasoner

# Run retrieval
python run_retrieval.py --algo_name RGS --dataset_name psychology --model_name bge-reasoner
```

## Requirements

### GPU (Recommended)
- **BGE-Reasoner**: A100/V100 (24GB+), ~2-4h per dataset
- **BGE-Large**: RTX 3090 (8GB+), ~30min per dataset

### Cloud Options
- **Lambda Labs**: A100 40GB @ $1.10/hr → ~$40 total
- **Colab Pro+**: A100 @ $50/month

## Dataset

**BRIGHT**: 12 subsets (biology, earth_science, economics, psychology, robotics, stackoverflow, sustainable_living, pony, leetcode, aops, theoremqa_theorems, theoremqa_questions)

**Total**: 1,384 queries + 700K documents

## Models

| Model | Params | GPU | Use Case |
|-------|--------|-----|----------|
| bge-reasoner | 8B | 24GB | Target (NDCG~37.1%) |
| bge-large | 335M | 8GB | Baseline |
| bge-base | 110M | 4GB | Fast baseline |
| bge-small | 33M | 2GB | Quick test |

## Structure

```
├── config.yaml              # Model & dataset config
├── generate_embeddings.py   # Embedding generation
├── run_retrieval.py         # Retrieval experiments
├── run_retrieval_single.py  # Single query retrieval
└── requirements.txt         # Dependencies
```

## Expected Results

```
Target (BGE-Reasoner): NDCG@10 = 37.1%
Baseline (BGE-Large):  NDCG@10 ≈ 30%
Baseline (BM25):       NDCG@10 ≈ 18%
```

## References

- [BGE-Reasoner Model](https://huggingface.co/BAAI/bge-reasoner-embed-qwen3-8b-0923)
- [BRIGHT Benchmark](https://brightbenchmark.github.io/)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- [Original RGS Paper](https://github.com/xuhaike/Reranker-Guided-Search)

## License

MIT
