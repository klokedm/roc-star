---
language:
  - en
license: apache-2.0
task_categories:
  - text-classification
task_ids:
  - sentiment-classification
pretty_name: Roc-star Extended – Twitter Sentiment (tokenized + embeddings)
size_categories:
  - 1M<n<10M
tags:
  - roc-star
  - auc-optimization
  - sentiment-analysis
  - twitter
  - extended
dataset_info:
  features:
    - name: tokens
      sequence: int32
      description: Pre-tokenized word-index sequence (max length 30)
    - name: label
      dtype: float32
      description: Sentiment label – 1.0 positive, 0.0 negative
  splits:
    - name: train
      num_examples: 1280000
    - name: validation
      num_examples: 320000
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
      - split: validation
        path: data/validation-*.parquet
---

# Roc-star Extended – Twitter Sentiment Dataset

This dataset accompanies the
[klokedm/roc-star](https://github.com/klokedm/roc-star) repository and the
`roc_star_loss` / `epoch_update_gamma` loss functions described in:

> C. Reiss, *"Roc-star: An objective function for ROC-AUC that actually works"*,
> based on Yan et al. (2003) "Optimizing Classifier Performance via an
> Approximation to the Wilcoxon-Mann-Whitney Statistic."

## What makes this the *extended* repository?

| Feature | Non-extended | Extended (this repo) |
|---------|--------------|----------------------|
| Source code (`rocstar.py`) | ✓ | ✓ |
| Raw tweet text | – | – (not redistributed) |
| **Pre-tokenised sequences** | – | **✓** |
| **GloVe embedding matrix** | – | **✓** |
| Train / validation splits | – | **✓** |
| HF data viewer config | – | **✓** |

The pre-tokenised and pre-embedded data let users reproduce training with
[`example.py`](https://github.com/klokedm/roc-star/blob/master/example.py)
without re-running the full preprocessing pipeline.

## Dataset structure

```
data/
  train-00000-of-NNNNN.parquet        # training split  (tokens, label)
  validation-00000-of-NNNNN.parquet   # validation split (tokens, label)
embeddings/
  embedding.npy                        # GloVe embedding matrix (vocab_size × 300)
```

### Features

| Column | Dtype | Description |
|--------|-------|-------------|
| `tokens` | `Sequence(int32)` | Word-index token sequence (length ≤ 30) |
| `label` | `float32` | 1.0 = positive sentiment, 0.0 = negative |

## Usage

```python
from datasets import load_dataset
import numpy as np

# Load tokenised splits
ds = load_dataset("klokedm/roc-star-extended")
train = ds["train"]
val   = ds["validation"]

# Load the embedding matrix
from huggingface_hub import hf_hub_download
emb_path = hf_hub_download(
    repo_id="klokedm/roc-star-extended",
    filename="embeddings/embedding.npy",
    repo_type="dataset",
)
embedding_matrix = np.load(emb_path)
```

## Source

Original data: Sentiment140 Twitter corpus (~1.6 M tweets).  
Preprocessing: Standard GloVe tokenisation, max sequence length 30.  
Train/validation split: 80 % / 20 % random split.  
Preprocessed data originally hosted at
`allegro-datasets.s3.amazonaws.com/roc_star_data/`.

## Citation

```bibtex
@misc{reiss2020rocstar,
  author       = {Reiss, Christopher},
  title        = {Roc-star: An objective function for ROC-AUC that actually works},
  year         = {2020},
  howpublished = {\url{https://github.com/klokedm/roc-star}}
}

@inproceedings{yan2003optimizing,
  author    = {Yan, Lian and Dodier, Robert and Mozer, Michael C. and Wolniewicz, Richard},
  title     = {Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic},
  booktitle = {Proceedings of the 20th International Conference on Machine Learning (ICML-03)},
  year      = {2003}
}
```

## License

Apache 2.0
