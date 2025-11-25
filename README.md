# BGE-Reasoner BRIGHT Benchmark Evaluation

## Overview
Evaluating BGE-Reasoner model on the BRIGHT benchmark using the RGS (Retrieval-Guided Search) two-stage algorithm:
1. Stage 1: FAISS dense retrieval with pre-computed BGE-Reasoner embeddings
2. Stage 2: GPT-4o-mini LLM Reranking

## Files
- `run_retrieval.py` - Main retrieval script with RGS algorithm
- `run_retrieval_single.py` - Single dataset retrieval script
- `generate_embeddings.py` - Generate BGE-Reasoner embeddings
- `fix_faiss.py` - FAISS index utility

## Usage
```bash
# Set OpenAI API Key
export OPENAI_API_KEY="your-api-key"

# Run single dataset
python run_retrieval.py \
    --benchmark_name BRIGHT \
    --dataset_name biology \
    --model_name bge-reasoner \
    --algo_name RGS \
    --embedding_path ./embedding_data

# Run in background
nohup python -u run_retrieval.py \
    --benchmark_name BRIGHT \
    --dataset_name biology \
    --model_name bge-reasoner \
    --algo_name RGS \
    --embedding_path ./embedding_data > logs/biology.log 2>&1 &
```

## Datasets (BRIGHT benchmark)
- biology, earth_science, economics, psychology, robotics
- stackoverflow, sustainable_living, pony
- leetcode, aops
- theoremqa_theorems, theoremqa_questions

## Dependencies
- faiss-cpu
- numpy
- openai
- pytrec_eval
- transformers

## Directory Structure
```
BGE-Reasoner-BRIGHT/
├── embedding_data/BRIGHT/     # Pre-computed embeddings
├── indices/                   # FAISS index files
├── logs/                      # Run logs
├── results/                   # Output results
└── *.py                       # Code files
```

## Results (11/12 datasets)
| Dataset | NDCG@1 | NDCG@5 | NDCG@10 |
|---------|--------|--------|---------|
| biology | 0.359 | 0.308 | 0.343 |
| earth_science | 0.448 | 0.368 | 0.400 |
| economics | 0.184 | 0.199 | 0.228 |
| pony | 0.313 | 0.201 | 0.179 |
| psychology | 0.277 | 0.271 | 0.296 |
| robotics | 0.208 | 0.191 | 0.209 |
| stackoverflow | 0.222 | 0.244 | 0.276 |
| sustainable_living | 0.213 | 0.207 | 0.231 |
| theoremqa_theorems | 0.329 | 0.406 | 0.455 |
| theoremqa_questions | 0.314 | 0.323 | 0.337 |
| aops | 0.063 | 0.043 | 0.060 |
| leetcode | - | - | - (pending) |
