import argparse
import os
import sys
import time
import yaml
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_embeddings(dataset_name, model_name, config, force=False):
    print("="*70)
    print(f"Dataset: {dataset_name} | Model: {model_name}")
    print("="*70)

    if model_name not in config['embedding_models']:
        raise ValueError(f"Model '{model_name}' not found in config")

    model_config = config['embedding_models'][model_name]
    save_path = config['experiment']['embedding_path']
    os.makedirs(save_path, exist_ok=True)

    print(f"\nLoading BRIGHT {dataset_name} dataset...")
    try:
        data = load_dataset('xlangai/BRIGHT', 'examples', keep_in_memory=False)[dataset_name]
        documents = load_dataset('xlangai/BRIGHT', 'documents', keep_in_memory=False)[dataset_name]
        print(f"✓ Loaded {len(data)} queries, {len(documents)} documents")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

    queries = {q["id"]: q["query"] for q in data}
    corpus = {doc["id"]: {"title": "", "text": doc["content"]} for doc in documents}

    print(f"\nLoading model: {model_config['model_id']}")
    device = model_config.get('device', 'cpu')
    if device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            device = 'cpu'

    try:
        model = SentenceTransformer(model_config['model_id'], device=device)
        print(f"Model loaded on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    print(f"\n{'='*70}")
    print("PHASE 1: Encoding Documents")
    print(f"{'='*70}")

    passage_file = os.path.join(save_path, f"{dataset_name}_{model_name}_passage_embeddings.npy")

    if os.path.exists(passage_file) and not force:
        print(f"File exists: {passage_file}")
    else:
        passages = [v["title"] + " " + v["text"] for v in corpus.values()]
        print(f"Encoding {len(passages)} passages...")

        batch_size = model_config.get('batch_size', 32)
        embeddings_list = []

        for i in tqdm(range(0, len(passages), batch_size), desc="Encoding"):
            batch = passages[i:min(len(passages), i+batch_size)]
            emb = model.encode(
                batch,
                batch_size=batch_size,  # Use config value instead of hardcoded 32
                normalize_embeddings=model_config.get('normalize', True),
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings_list.append(emb)

        passage_embeddings = np.vstack(embeddings_list)
        np.save(passage_file, passage_embeddings)
        file_size = os.path.getsize(passage_file) / 1024 / 1024
        print(f"Saved: {passage_file} ({passage_embeddings.shape}, {file_size:.1f} MB)")

    print(f"\n{'='*70}")
    print("PHASE 2: Encoding Queries")
    print(f"{'='*70}")

    query_file = os.path.join(save_path, f"{dataset_name}_test_{model_name}_query_embeddings.npy")

    if os.path.exists(query_file) and not force:
        print(f"File exists: {query_file}")
    else:
        query_list = list(queries.values())

        if 'bge' in model_name and dataset_name in config['dataset_prompts']:
            prompt = config['dataset_prompts'][dataset_name]
            query_list = [prompt + q for q in query_list]

        print(f"Encoding {len(query_list)} queries...")
        query_batch_size = model_config.get('batch_size', 32)
        query_embeddings = model.encode(
            query_list,
            batch_size=query_batch_size,  # Use config value instead of hardcoded 32
            normalize_embeddings=model_config.get('normalize', True),
            show_progress_bar=True,
            convert_to_numpy=True
        )

        np.save(query_file, query_embeddings)
        print(f"Saved: {query_file} ({query_embeddings.shape})")

    print(f"\n{'='*70}\nCompleted: {dataset_name}\n{'='*70}\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for BRIGHT benchmark datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset name or 'all' for all datasets"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bge-reasoner",
        help="Model name (bge-reasoner, bge-large, bge-base, bge-small)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if embeddings exist"
    )

    args = parser.parse_args()

    config = load_config(args.config)
    datasets = config['bright_datasets'] if args.dataset == "all" else [args.dataset]

    start_time = time.time()
    successful, failed = 0, 0

    print(f"\nBGE-Reasoner Embedding Generation\nModel: {args.model}\nDatasets: {len(datasets)}\n")

    for i, dataset in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] Processing {dataset}...")

        try:
            success = generate_embeddings(dataset, args.model, config, args.force)
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error: {e}")
            failed += 1

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*70}\nSummary: {successful} successful, {failed} failed\nTime: {elapsed:.1f} min\n{'='*70}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
