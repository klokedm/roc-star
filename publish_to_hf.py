"""
publish_to_hf.py - Publish the roc-star extended dataset repository to Hugging Face Hub.

This script publishes the roc-star extended dataset to HuggingFace Hub, including:
  - Pre-tokenized training and validation splits (train/validation Parquet files)
  - Pre-computed GloVe embeddings (stored as a separate data file)
  - Dataset card (README.md) with YAML metadata for the HF data viewer
  - Dataset configuration YAML for the data viewer

The "extended" repository differs from the non-extended (code-only) repository in that it
includes the pre-processed data artefacts that example.py needs, so users can reproduce
training without re-running the expensive preprocessing pipeline.

Usage:
    HF_TOKEN=<your_token> python publish_to_hf.py [--repo-id REPO_ID] [--dry-run]

Requirements:
    pip install huggingface_hub datasets pyarrow numpy requests

Environment variables:
    HF_TOKEN : HuggingFace write token (required unless --dry-run is used)
"""

import argparse
import io
import os
import pickle
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

try:
    from huggingface_hub import HfApi, DatasetCard, DatasetCardData
    from datasets import Dataset, DatasetDict, Features, Sequence, Value
except ImportError:
    print("ERROR: Required packages missing. Run:")
    print("  pip install huggingface_hub datasets pyarrow numpy requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_REPO_ID = "klokedm/roc-star-extended"

# Parquet shard filename pattern (single-shard – no sharding needed for this dataset size)
TRAIN_PARQUET = "data/train-00000-of-00001.parquet"
VALIDATION_PARQUET = "data/validation-00000-of-00001.parquet"

# Source data URLs (Allegro public S3 bucket used by the original example.py)
TOKENIZED_URL = (
    "https://allegro-datasets.s3.amazonaws.com/roc_star_data/tokenized.pkl.zip"
)
EMBEDDING_URL = (
    "https://allegro-datasets.s3.amazonaws.com/roc_star_data/embedding.pkl.zip"
)

# Dataset viewer config filename (placed in repo root, next to README.md)
DATASET_CONFIG_FILENAME = "dataset_config.yaml"

# Sibling files (same directory as this script) used as authoritative sources.
_SCRIPT_DIR = Path(__file__).resolve().parent
_CARD_SRC = _SCRIPT_DIR / "hf_dataset_card.md"
_CFG_SRC = _SCRIPT_DIR / DATASET_CONFIG_FILENAME

# ---------------------------------------------------------------------------
# Dataset card content (README.md for the HF repo)
# Mirrors the structure of the non-extended repo's card but adds extended fields.
# Read from hf_dataset_card.md when available; inline fallback keeps the script
# self-contained.
# ---------------------------------------------------------------------------

DATASET_CARD_CONTENT = """\
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
  train-00000-of-NNNNN.parquet   # training split (token sequences + labels)
  validation-00000-of-NNNNN.parquet
embeddings/
  embedding.npy                  # GloVe embedding matrix  (vocab_size × 300)
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

## License

Apache 2.0
"""

# ---------------------------------------------------------------------------
# Dataset viewer config YAML
# This file configures the HF dataset viewer columns and splits.
# ---------------------------------------------------------------------------

DATASET_CONFIG_YAML = """\
# dataset_config.yaml
# Hugging Face dataset viewer configuration for klokedm/roc-star-extended

configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
      - split: validation
        path: data/validation-*.parquet
    features:
      tokens:
        dtype: sequence
        feature:
          dtype: int32
      label:
        dtype: float32
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download_and_unpickle(url: str, tmp_dir: Path, name: str):
    """Download a zip of a pickle from *url* and return the unpickled object."""
    zip_path = tmp_dir / f"{name}.zip"
    print(f"  Downloading {url} …")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(zip_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
    print(f"  Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path) as zf:
        pkl_name = next(n for n in zf.namelist() if n.endswith(".pkl"))
        with zf.open(pkl_name) as pkl_fh:
            obj = pickle.load(pkl_fh)
    return obj


def _load_local_pickle(path: Path):
    """Load a pickle file from a local path."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _arrays_to_parquet(tokens_list, labels_array, out_path: Path) -> None:
    """Write (tokens, label) arrays to a Parquet file."""
    # Build an Arrow table with a list<int32> column for tokens
    token_type = pa.list_(pa.int32())
    token_col = pa.array(
        [np.array(t, dtype=np.int32).tolist() for t in tokens_list],
        type=token_type,
    )
    label_col = pa.array(labels_array.astype(np.float32), type=pa.float32())
    table = pa.table({"tokens": token_col, "label": label_col})
    pq.write_table(table, str(out_path))
    print(f"  Written {len(labels_array):,} rows → {out_path}")


# ---------------------------------------------------------------------------
# Main publish logic
# ---------------------------------------------------------------------------


def build_local_dataset(tmp_dir: Path, args) -> None:
    """Download / load data and write local Parquet + embedding files."""

    data_dir = tmp_dir / "data"
    data_dir.mkdir()
    emb_dir = tmp_dir / "embeddings"
    emb_dir.mkdir()

    # --- Tokenised data ---
    if args.tokenized_pkl:
        print("Loading tokenised data from local file …")
        payload = _load_local_pickle(Path(args.tokenized_pkl))
    else:
        payload = _download_and_unpickle(TOKENIZED_URL, tmp_dir, "tokenized")

    x_train, x_valid, y_train, y_valid = payload

    # Optionally truncate for testing
    if args.trunc and args.trunc > 0:
        print(f"  Truncating to first {args.trunc} training samples (--trunc flag)")
        x_train = x_train[: args.trunc]
        y_train = y_train[: args.trunc]

    print("Writing train split …")
    _arrays_to_parquet(x_train, np.array(y_train), tmp_dir / TRAIN_PARQUET)
    print("Writing validation split …")
    _arrays_to_parquet(x_valid, np.array(y_valid), tmp_dir / VALIDATION_PARQUET)

    # --- Embedding matrix ---
    if args.embedding_pkl:
        print("Loading embedding matrix from local file …")
        embedding_matrix = _load_local_pickle(Path(args.embedding_pkl))
    else:
        embedding_matrix = _download_and_unpickle(EMBEDDING_URL, tmp_dir, "embedding")

    emb_path = emb_dir / "embedding.npy"
    # Use .astype() rather than np.array() to avoid redundant copy when already ndarray
    np.save(str(emb_path), np.asarray(embedding_matrix, dtype=np.float32))
    print(f"  Embedding matrix shape: {np.asarray(embedding_matrix).shape} → {emb_path}")

    # --- Dataset card ---
    card_path = tmp_dir / "README.md"
    card_content = _CARD_SRC.read_text(encoding="utf-8") if _CARD_SRC.exists() else DATASET_CARD_CONTENT
    card_path.write_text(card_content, encoding="utf-8")
    print(f"  Written {card_path}")

    # --- Dataset viewer config ---
    cfg_path = tmp_dir / DATASET_CONFIG_FILENAME
    cfg_content = _CFG_SRC.read_text(encoding="utf-8") if _CFG_SRC.exists() else DATASET_CONFIG_YAML
    cfg_path.write_text(cfg_content, encoding="utf-8")
    print(f"  Written {cfg_path}")


def publish(tmp_dir: Path, args) -> None:
    """Upload the local dataset folder to HuggingFace Hub."""
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print(
            "ERROR: No HuggingFace token found.\n"
            "  Set HF_TOKEN environment variable or pass --token."
        )
        sys.exit(1)

    api = HfApi(token=token)

    repo_id = args.repo_id
    print(f"\nCreating / verifying dataset repo: {repo_id} …")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=args.private,
    )

    print(f"Uploading files from {tmp_dir} …")
    api.upload_folder(
        folder_path=str(tmp_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Publish extended roc-star dataset (tokenized splits + embeddings)",
        ignore_patterns=["*.zip"],
    )
    print(f"\n✅ Published successfully: https://huggingface.co/datasets/{repo_id}")


def verify(args) -> None:
    """Verify that the published repo contains the expected files."""
    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    repo_id = args.repo_id
    print(f"\nVerifying published repo: {repo_id} …")
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:
        print(f"ERROR: Could not list files in {repo_id}: {exc}")
        sys.exit(1)

    files = list(files)
    print(f"  Files found ({len(files)}):")
    for f in sorted(files):
        print(f"    {f}")

    required = {
        "README.md",
        DATASET_CONFIG_FILENAME,
    }
    required_prefixes = ("data/train-", "data/validation-", "embeddings/embedding")

    missing = []
    for req in required:
        if req not in files:
            missing.append(req)
    for prefix in required_prefixes:
        if not any(f.startswith(prefix) for f in files):
            missing.append(f"<file matching {prefix}*>")

    if missing:
        print(f"\n❌ Verification FAILED – missing files: {missing}")
        sys.exit(1)
    else:
        print("\n✅ Verification PASSED – all required files are present.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Publish the roc-star extended dataset to HuggingFace Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="HuggingFace dataset repo ID (e.g. username/repo-name)",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HuggingFace write token (or set HF_TOKEN env var)",
    )
    p.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create the repo as private",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Build the local dataset but skip the actual upload to HuggingFace",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        default=False,
        help="Skip build and upload; only verify an already-published repo",
    )
    p.add_argument(
        "--tokenized-pkl",
        default=None,
        metavar="PATH",
        help="Path to a local tokenized.pkl (skips S3 download)",
    )
    p.add_argument(
        "--embedding-pkl",
        default=None,
        metavar="PATH",
        help="Path to a local embedding.pkl (skips S3 download)",
    )
    p.add_argument(
        "--trunc",
        type=int,
        default=None,
        metavar="N",
        help="Truncate training set to first N samples (useful for testing)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.verify_only:
        verify(args)
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        print("=== Step 1: Build local dataset ===")
        build_local_dataset(tmp_dir, args)

        if args.dry_run:
            print("\n[--dry-run] Skipping upload. Local files:")
            for p in sorted(tmp_dir.rglob("*")):
                if p.is_file():
                    size_kb = p.stat().st_size / 1024
                    print(f"  {p.relative_to(tmp_dir)}  ({size_kb:.1f} kB)")
            print("\n[--dry-run] No files were uploaded to HuggingFace.")
            return

        print("\n=== Step 2: Publish to HuggingFace Hub ===")
        publish(tmp_dir, args)

        print("\n=== Step 3: Verify publication ===")
        verify(args)


if __name__ == "__main__":
    main()
