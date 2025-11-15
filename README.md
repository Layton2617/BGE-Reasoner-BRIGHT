# BGE-Reasoner BRIGHT Embeddings Generator

Generate high-quality embeddings for the BRIGHT benchmark using BGE-Reasoner, a state-of-the-art 8B parameter embedding model.

## Features

- Support for all 12 BRIGHT benchmark datasets
- Efficient batch processing with GPU support
- Pre-generated embeddings for the `pony` dataset included
- Configurable batch sizes and model parameters
- Progress tracking with automatic checkpointing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bge-reasoner-bright-embeddings
cd bge-reasoner-bright-embeddings

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Generate Embeddings

Generate embeddings for a single dataset:

```bash
python generate_embeddings.py --dataset biology --model bge-reasoner
```

Generate embeddings for all datasets:

```bash
python generate_embeddings.py --dataset all --model bge-reasoner
```

### Batch Generation Script

Use the included shell script to generate all remaining datasets:

```bash
chmod +x generate_all_datasets.sh
./generate_all_datasets.sh
```

## Available Datasets

| Dataset | Domain | Documents | Queries | Est. Time |
|---------|--------|-----------|---------|-----------|
| pony | General Knowledge | 7,894 | 112 | ~24 min |
| biology | Biology | ~50K | ~100 | ~30 min |
| earth_science | Earth Science | ~30K | ~80 | ~25 min |
| economics | Economics | ~35K | ~90 | ~25 min |
| psychology | Psychology | ~30K | ~85 | ~25 min |
| robotics | Robotics | ~25K | ~70 | ~20 min |
| stackoverflow | Programming | ~100K | ~200 | ~45 min |
| sustainable_living | Sustainability | ~20K | ~60 | ~20 min |
| leetcode | Algorithms | ~40K | ~120 | ~30 min |
| aops | Math Competitions | ~35K | ~100 | ~25 min |
| theoremqa_theorems | Math Theorems | ~25K | ~75 | ~20 min |
| theoremqa_questions | Math Problems | ~25K | ~75 | ~20 min |

## Configuration

Edit `config.yaml` to customize:
- Model parameters
- Batch sizes
- Device settings (CPU/GPU)
- Output paths

## Output Format

Embeddings are saved as NumPy arrays (.npy) in the following structure:

```
embedding_data/BRIGHT/
├── [dataset]_bge-reasoner_passage_embeddings.npy  # Document embeddings
└── [dataset]_test_bge-reasoner_query_embeddings.npy  # Query embeddings
```

Each embedding is a 4096-dimensional vector normalized to unit length.

## System Requirements

- **GPU Memory**: 30GB+ recommended (H100/A100)
- **Disk Space**: ~2GB per dataset
- **Python**: 3.8+

## Pre-generated Embeddings

The `pony` dataset embeddings are included and have been validated with:
- NDCG@10: 19.93% using FAISS
- Dimensions: 4096
- Documents: 7,894
- Queries: 112

## Verification

Verify embedding quality with simple FAISS test:

```python
import numpy as np
import faiss

# Load embeddings
query_emb = np.load('embedding_data/BRIGHT/pony_test_bge-reasoner_query_embeddings.npy')
passage_emb = np.load('embedding_data/BRIGHT/pony_bge-reasoner_passage_embeddings.npy')

# Build FAISS index
index = faiss.IndexFlatIP(passage_emb.shape[1])
index.add(passage_emb)

# Test retrieval
k = 10
distances, indices = index.search(query_emb[0:1], k)
print(f"Top {k} results: {indices[0]}")
```

## Model Information

BGE-Reasoner (`BAAI/bge-reasoner-embed-qwen3-8b-0923`):
- Architecture: 8B parameters
- Context: Extended context support
- Performance: State-of-the-art on BRIGHT benchmark
- Download: Automatically downloaded from Hugging Face on first use

## Citation

If you use these embeddings in your research, please cite:

```bibtex
@misc{bge-reasoner-bright,
  title={BGE-Reasoner BRIGHT Embeddings},
  author={Your Name},
  year={2024},
  howpublished={GitHub}
}
```

## License

MIT License - See LICENSE file for details

## Contact

For issues or questions, please open an issue on GitHub.