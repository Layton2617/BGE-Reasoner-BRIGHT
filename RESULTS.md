# Detailed Experiment Results

## Overview

Full evaluation of BGE-Reasoner on all 12 BRIGHT benchmark datasets using RGS (Retrieval-Guided Search) with GPT-4o-mini reranking.

## Results by Dataset

### Biology
| Algorithm | NDCG@1 | NDCG@5 | NDCG@10 | API Calls |
|-----------|--------|--------|---------|-----------|
| Retrieve (Baseline) | 19.4% | 17.9% | 21.2% | 0 |
| RGS (quota=100) | 43.7% | 40.1% | **44.4%** | 36 |
| RGS (quota=300) | 47.6% | 40.5% | 44.1% | 79 |

### Psychology
| Algorithm | NDCG@1 | NDCG@5 | NDCG@10 | API Calls |
|-----------|--------|--------|---------|-----------|
| Retrieve (Baseline) | 19.8% | 23.2% | 25.2% | 0 |
| RGS (quota=100) | 35.6% | 40.1% | **43.5%** | 36 |
| RGS (quota=300) | 36.6% | 39.2% | 42.7% | 86 |
| RGS (quota=500) | 37.6% | 39.5% | 41.1% | 145 |

### Sustainable Living
| Algorithm | NDCG@1 | NDCG@5 | NDCG@10 | API Calls |
|-----------|--------|--------|---------|-----------|
| Retrieve (Baseline) | 15.7% | 16.0% | 18.3% | 0 |
| RGS (quota=100) | 30.6% | 29.0% | 31.9% | 37 |
| RGS (quota=300) | 34.3% | 32.3% | **34.8%** | 86 |
| RGS (quota=500) | 33.3% | 32.5% | 34.3% | 149 |

### TheoremQA Theorems
| Algorithm | NDCG@1 | NDCG@5 | NDCG@10 | API Calls |
|-----------|--------|--------|---------|-----------|
| Retrieve (Baseline) | 30.3% | 35.8% | 41.7% | 0 |
| RGS (quota=100) | 35.5% | 40.8% | 45.6% | 32 |
| RGS (quota=300) | 31.6% | 41.8% | 45.4% | 78 |
| RGS (quota=500) | 34.2% | 41.4% | **46.8%** | 122 |

### Earth Science
| Algorithm | NDCG@1 | NDCG@5 | NDCG@10 | API Calls |
|-----------|--------|--------|---------|-----------|
| Retrieve (Baseline) | 19.8% | 22.5% | 24.8% | 0 |
| RGS (quota=100) | 47.4% | 39.6% | **41.2%** | 34 |
| RGS (quota=300) | 44.8% | 37.3% | 40.1% | 88 |
| RGS (quota=500) | 43.1% | 35.7% | 39.2% | 150 |

### Economics
| Algorithm | NDCG@1 | NDCG@5 | NDCG@10 | API Calls |
|-----------|--------|--------|---------|-----------|
| Retrieve (Baseline) | 14.6% | 15.8% | 17.5% | 0 |
| RGS (quota=100) | 22.3% | 23.7% | 24.2% | 37 |
| RGS (quota=300) | 23.3% | 24.9% | **25.8%** | 78 |
| RGS (quota=500) | 16.5% | 20.0% | 22.1% | 59 |

### TheoremQA Questions
| Algorithm | NDCG@1 | NDCG@5 | NDCG@10 | API Calls |
|-----------|--------|--------|---------|-----------|
| Retrieve (Baseline) | 25.8% | 26.7% | 27.4% | 0 |
| RGS (quota=100) | 30.9% | 31.8% | **33.1%** | 0 |
| RGS (quota=300) | 30.9% | 31.8% | 33.1% | 0 |
| RGS (quota=500) | 30.9% | 31.8% | 33.1% | 0 |

### StackOverflow
| Algorithm | NDCG@1 | NDCG@5 | NDCG@10 | API Calls |
|-----------|--------|--------|---------|-----------|
| Retrieve (Baseline) | 19.7% | 20.8% | 22.1% | 0 |
| RGS (quota=100) | 23.1% | 24.4% | **27.6%** | 0 |
| RGS (quota=300) | 23.1% | 24.4% | 27.6% | 0 |
| RGS (quota=500) | 23.1% | 24.4% | 27.6% | 0 |

### LeetCode
| Algorithm | NDCG@1 | NDCG@5 | NDCG@10 | API Calls |
|-----------|--------|--------|---------|-----------|
| Retrieve (Baseline) | 4.2% | 15.5% | 18.9% | 0 |
| RGS (quota=100) | 4.9% | 17.0% | **21.7%** | 0 |
| RGS (quota=300) | 4.9% | 17.0% | 21.7% | 0 |
| RGS (quota=500) | 4.9% | 17.0% | 21.7% | 0 |

### Robotics
| Algorithm | NDCG@1 | NDCG@5 | NDCG@10 | API Calls |
|-----------|--------|--------|---------|-----------|
| Retrieve (Baseline) | 12.9% | 15.5% | 16.2% | 0 |
| RGS (quota=100) | 14.9% | 17.8% | **19.0%** | 0 |
| RGS (quota=300) | 14.9% | 17.8% | 19.0% | 0 |
| RGS (quota=500) | 14.9% | 17.8% | 19.0% | 0 |

### Pony
| Algorithm | NDCG@1 | NDCG@5 | NDCG@10 | API Calls |
|-----------|--------|--------|---------|-----------|
| Retrieve (Baseline) | 11.6% | 12.3% | 12.5% | 0 |
| RGS (quota=100) | 14.3% | 14.9% | **13.9%** | 0 |
| RGS (quota=300) | 14.3% | 14.9% | 13.9% | 0 |
| RGS (quota=500) | 14.3% | 14.9% | 13.9% | 0 |

### AOPS
| Algorithm | NDCG@1 | NDCG@5 | NDCG@10 | API Calls |
|-----------|--------|--------|---------|-----------|
| Retrieve (Baseline) | 2.7% | 2.8% | 3.2% | 0 |
| RGS (quota=100) | 3.6% | 3.1% | 3.7% | 33 |
| RGS (quota=300) | 1.8% | 2.8% | 4.2% | 91 |
| RGS (quota=500) | 4.5% | 3.8% | **4.7%** | 156 |

## Key Findings

1. **Best Overall Performance**: Biology (44.4%) and TheoremQA Theorems (46.8%)
2. **Largest Improvement**: Biology achieved 2.1x improvement over baseline
3. **RGS Efficiency**: quota=100 often achieves near-optimal results with fewer API calls
4. **Domain Variation**: Performance varies significantly across domains (4.7% - 46.8%)

## Experimental Setup

- **Hardware**: Lambda Labs H100 80GB
- **Embedding Model**: BAAI/bge-reasoner-embed-qwen3-8b-0923
- **Reranking Model**: GPT-4o-mini
- **Total Runtime**: ~4 hours for all 12 datasets
