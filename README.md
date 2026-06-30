# Retrieval-Augmented Few-Shot Malware Detection (Replication)

A replication of *Retrieval-Augmented Few-Shot Malware Detection via Binary
Visualization and Vision-Language Embeddings* (Jung, Kwak & Lee, *Appl. Sci.*
2026, 16, 4100). The method classifies malware families from binary-as-image
representations using a **frozen** Vision-Language Model as a feature extractor
plus similarity-based retrieval, with **no task-specific training**.

## Method Overview

The pipeline decouples representation from adaptation:

1. **Binary-to-image**: malware binaries are rendered as grayscale images
   (handled upstream; MalImg ships pre-rendered).
2. **Frozen embedding**: each image is passed through Qwen2.5-VL-3B and reduced
   to a single 2048-dim vector. Adaptation to new families is just appending
   vectors to an external store, never updating weights.
3. **Retrieval classification**: a query embedding retrieves its top-k nearest
   support embeddings by cosine similarity (FAISS), and the predicted family is
   chosen by similarity-weighted voting.

## Pipeline Stages

### 1. Data loading (`dataloader.py`)
Builds a `{family_name: [image_paths]}` dictionary by walking the dataset root,
keyed on subfolder names. Images are resized to 384x384 and converted from
single-channel grayscale to 3-channel RGB lazily, at access time (no resized
copies are written to disk).

### 2. Embedding extraction (`main.py`)
Loads frozen Qwen2.5-VL-3B (`eval` mode, `inference_mode` forward passes).
For each image: resize, run through the processor (empty text prompt, image-only),
forward pass with `output_hidden_states=True`. The 2048-dim embedding is formed by:
- averaging the **last four LM hidden layers**,
- **mean-pooling over the token axis**,
- **L2-normalizing** the result.

All embeddings are cached once to `malimg_embeddings.pt` as
`{family: [n, 2048] tensor}`. This is the expensive, one-time GPU stage.

### 3. Retrieval and evaluation (`LM_part.py`)
- Loads the cached embeddings.
- Per episode: samples N families and K+5 vectors per family (K support, 5 query,
  drawn disjointly).
- Builds a fresh `faiss.IndexFlatIP` over the episode's support vectors
  (cosine similarity via normalized embeddings + inner product).
- Searches each query for its nearest support neighbors, then predicts via
  similarity-weighted voting (sum of similarity scores per family, argmax).
- Accuracy is averaged over many episodes.

## Evaluation Protocol

N-way K-shot episodic protocol: **N = 5**, **K in {1, 3, 5, 10}**, 5 query
samples per class, search-k = 10, results averaged over multiple episodes.

## Results (in progress)

| Shot (K) | This replication | Paper |
|----------|------------------|-------|
| 1-shot   | 0.742            | 0.878 |
| 3-shot   | 0.802            | 0.875 |
| 5-shot   | 0.768            | 0.864 |
| 10-shot  | 0.872            | 0.886 |

The 10-shot result tracks the paper closely. The 5-shot point currently sits
below the reported value; see Known Deviations.

## Known Deviations and Assumptions

The paper underspecifies several steps. The following choices were made and may
explain gaps from the reported numbers:

- **Which hidden states**: the paper says "weighted aggregation of the last four
  hidden layers" without specifying vision tower vs. LM, the exact layers, or the
  weights. This replication uses the **last four LM decoder layers** (the 2048
  dimension matches the 3B LM hidden size) with an **equal-weight average**.
- **Token pooling**: mean over **all tokens**, including structural/scaffolding
  tokens, rather than masking to image-token positions only.
- **Figure inconsistencies in the source paper**: figures label a "position-aware
  adapter" as *trainable* and the vector DB as *approximate (ANN)*, both of which
  contradict the text (frozen encoder, exact `IndexFlatIP`). This replication
  follows the **text**: fully frozen, no adapter, exact search.
- **1-shot retrieval**: with N=5 and K=1, the support pool is only 5 vectors, so
  search-k=10 cannot return 10 neighbors. The vote skips empty (`-1`) slots, so
  effective retrieval at 1-shot is 5 neighbors regardless of the requested k.

## Hardware Notes

- Extraction was run locally on a 4GB laptop RTX 3050, which forces heavy
  CPU offloading of the 3B model (full pass took several hours).
- **Recommended**: run extraction on a GPU with enough VRAM to hold the model
  with no offload (24GB+), which reduces the full MalImg pass to minutes.
- Retrieval and evaluation are CPU-only (`faiss-cpu`) and fast; they operate on
  the cached embeddings and never load the model.

## Dependencies

- `transformers`, `qwen-vl-utils`, `torch` (CUDA build for extraction)
- `faiss-cpu`
- `Pillow`
- `tqdm`

## Dataset

MalImg: 9,339 grayscale malware images across 25 families
(Nataraj et al., 2011). Expected layout: one subfolder per family, images inside.
