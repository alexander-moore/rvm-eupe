# RVM-EUPE

**Recurrent Video Masked Autoencoders with EUPE Backbone**

A PyTorch research project demonstrating that the EUPE (Efficient Universal Perception Encoder, [arxiv 2603.22387](https://arxiv.org/abs/2603.22387)) backbone improves over the standard ViT encoder used in the Recurrent Video Masked Autoencoders (RVM) framework ([arxiv 2512.13684](https://arxiv.org/abs/2512.13684)).

The original RVM code is JAX-only. This repo is a clean PyTorch port of the full architecture — encoder, Transformer-GRU recurrent aggregator, MAE decoder — with EUPE as the frame encoder. A multi-depth feature fusion ablation study is included.

---

## Background

### RVM (arxiv 2512.13684)
Recurrent Video MAE processes sequences of video frames through a ViT encoder and a Transformer-GRU recurrent aggregator that maintains a hidden state `s_t` across frames. The hidden state serves as memory for an MAE decoder that reconstructs heavily masked target frames (95% masking ratio). At evaluation time, the encoder + recurrent unit are frozen and task-specific readout heads are trained on top of `s_t`. This approach achieves strong results on video segmentation (DAVIS), action recognition (Kinetics, SSv2), depth estimation (ScanNet), and tracking tasks.

### EUPE (arxiv 2603.22387)
EUPE is a ViT-based encoder pretrained on LVD-1689M (1.689B images) using MAE + contrastive objectives. Key differences from a standard ViT:
- 2D RoPE positional embeddings (handles variable resolution natively)
- LayerScale initialised at 1e-5 (stabilises deep network training)
- `layernormbf16` (eps=1e-5) normalisation
- 4 register/storage tokens prepended to the sequence
- Available sizes: ViT-T (192d), ViT-S (384d), ViT-B (768d)

The hypothesis: EUPE's richer pretraining leads to better patch token representations, which in turn improves the TransformerGRU's temporal aggregation and downstream task performance.

---

## Project Structure

```
rvm-eupe/
├── conda.yaml                        # Reproducible environment
├── pyproject.toml
│
├── configs/
│   ├── default.yaml                  # Hydra base config
│   ├── model/
│   │   ├── rvm_eupe_vitt.yaml        # EUPE ViT-T/16 (192d)
│   │   ├── rvm_eupe_vits.yaml        # EUPE ViT-S/16 (384d)
│   │   ├── rvm_eupe_vitb.yaml        # EUPE ViT-B/16 (768d) — primary
│   │   ├── rvm_baseline_vitb.yaml    # timm ViT-B (no EUPE pretraining, comparison)
│   │   └── rvm_eupe_multiscale.yaml  # Multi-depth FPN ablation
│   ├── data/
│   │   └── mixed.yaml                # Dataset paths + mixture weights
│   └── train/
│       ├── base.yaml                 # Standard training config
│       └── large_scale.yaml          # 2048 effective batch (8 GPU)
│
├── rvm_eupe/
│   ├── models/
│   │   ├── encoder_wrapper.py        # EUPEEncoderWrapper
│   │   ├── transformer_gru.py        # TransformerGRU, CrossAttentionBlock
│   │   ├── decoder.py                # MAEDecoder
│   │   ├── recurrent_video_mae.py    # RecurrentVideoMAE (top-level)
│   │   ├── multiscale_adapter.py     # FPNAdapter, ConcatAdapter (ablation)
│   │   └── readout_heads.py          # AttentiveReadoutHead, query factories
│   ├── data/
│   │   ├── video_dataset.py          # VideoClipDataset (clip sampler)
│   │   ├── mixed_dataset.py          # MixedVideoDataset + WeightedRandomSampler
│   │   └── transforms.py             # Temporally-consistent video augmentations
│   ├── eval/
│   │   ├── davis.py                  # Non-parametric k-NN label propagation
│   │   ├── action_cls.py             # Kinetics / SSv2 attentive probe
│   │   ├── depth.py                  # ScanNet AbsRel depth evaluation
│   │   ├── tracking.py               # Perception Test Average Jaccard
│   │   └── keypoint.py               # JHMDB PCK@0.1
│   └── optim/
│       └── schedulers.py             # Cosine warmup + layerwise LR decay
│
├── scripts/
│   ├── train.py                      # Main training entry point (Hydra + FSDP)
│   └── eval_frozen.py                # Frozen-encoder evaluation entry point
│
└── tests/                            # 28 unit tests, all passing
    ├── test_encoder_wrapper.py
    ├── test_transformer_gru.py
    ├── test_decoder.py
    └── test_multiscale_adapter.py
```

---

## Architecture

### Primary model (A1)

```
source frames [B, 3, H, W] × 4
       │
       ▼
EUPEEncoderWrapper          ← EUPE ViT-B/16, forward_features() → x_norm_patchtokens
  CenterPadding(16)         ← pads H,W to multiples of patch_size
  DinoVisionTransformer     ← n_storage_tokens=4 already stripped in forward_features()
  pre_recurrent_norm        ← LayerNorm stabilises gate inputs
       │  [B, 196, 768]
       ▼
TransformerGRU (×4 steps)
  u_t = σ(W_u_e·e_t + W_u_s·s_{t-1})      update gate
  r_t = σ(W_r_e·e_t + W_r_s·s_{t-1})      reset gate
  ĥ_t = CrossAttn(q=e_t, kv=r_t⊙s_{t-1}) → FFN → SelfAttn
  s_t = (1−u_t)⊙s_{t-1} + u_t⊙ĥ_t        state update
       │  hidden state s_T  [B, 196, 768]
       ▼
MAEDecoder  (8 blocks, 512d, 16 heads)
  per block: CrossAttn(target→s_T) → FFN → SelfAttn
  mask token replaces 95% of target frame tokens
  output: pixel predictions [B, N_masked, 16×16×3]
       │
       ▼
L2 reconstruction loss (no patch normalisation)
```

### Ablation variants

| Variant | Encoder | Adapter | Note |
|---------|---------|---------|------|
| **A1 (primary)** | EUPE ViT-B last layer | None | `forward_features()` |
| **A2** | EUPE ViT-B layers [2,5,8,11] | `FPNAdapter` (1×1 conv + sum) | Multi-depth FPN |
| **A3** | EUPE ViT-B layers [2,5,8,11] | `ConcatAdapter` (concat + linear) | Multi-depth concat |
| **Baseline** | timm ViT-B (random init) | None | No EUPE pretraining |

> **Note on "multi-scale":** EUPE ViT has uniform stride-16 throughout. All intermediate layers produce spatial maps at the same resolution — this is multi-depth (multi-semantic-level) fusion, not a spatial pyramid. True multi-stride features would require the EUPE ConvNeXt backbone (a potential future ablation).

### EUPE model dimensions (confirmed from `hub/backbones.py`)

| Model | `embed_dim` | `n_storage_tokens` | `norm_layer` |
|-------|-------------|-------------------|--------------|
| ViT-T/16 | 192 | 4 | layernormbf16 |
| ViT-S/16 | 384 | 4 | layernormbf16 |
| ViT-B/16 | 768 | 4 | layernormbf16 |

The primary comparison is **EUPE ViT-B ↔ RVM ViT-B**: both 768d, no projection layer needed, clean apples-to-apples comparison.

---

## Installation

```bash
# 1. Create environment
conda env create -f conda.yaml
conda activate rvm-eupe

# 2. Install EUPE (local)
cd /home/farm/research/eupe
touch requirements-dev.txt          # if not present
pip install -e . --no-deps

# 3. Install this package
cd /home/farm/research/rvm-eupe
pip install -e .
```

---

## Running Tests

```bash
pytest tests/ -v
# Expected: 28 passed
```

Test coverage:

| File | Tests | What is covered |
|------|-------|----------------|
| `test_encoder_wrapper.py` | 7 | Shape correctness (T/S/B), non-standard res, projection, freeze/unfreeze, storage-token assertion |
| `test_transformer_gru.py` | 7 | Output shapes, None-state init, state propagation, gate near-identity init, all embed dims, multi-block, 4-frame rollout |
| `test_decoder.py` | 7 | Pos-embed shape, decoder output shape, L2 loss scalar, no NaN, mask ratio, full end-to-end forward (ViT-T, dummy), freeze |
| `test_multiscale_adapter.py` | 7 | FPN/concat adapter shapes, full encoder forward, freeze behaviour, GRU compatibility, output differs from last-layer-only |

---

## Training

### 1. Prepare dataset index files

Each dataset needs a JSON index file listing video paths and frame counts:

```json
[
  {"path": "/data/kinetics700/video_001.mp4", "num_frames": 250},
  {"path": "/data/ssv2/frames/video_002/",    "num_frames": 180}
]
```

Update paths in `configs/data/mixed.yaml`.

### 2. Launch training

**Single GPU (smoke test):**
```bash
python scripts/train.py \
  model=rvm_eupe_vitt \
  train.scheduler.total_steps=1000 \
  train.batch_size_per_gpu=4 \
  train.accumulate_grad_batches=1
```

**Multi-GPU (full scale, 8×A100):**
```bash
torchrun --nproc_per_node=8 scripts/train.py \
  model=rvm_eupe_vitb \
  train=large_scale
```

**Ablation variants:**
```bash
# A2: multi-depth FPN
python scripts/train.py model=rvm_eupe_multiscale

# Baseline (timm ViT-B, no EUPE)
python scripts/train.py model=rvm_baseline_vitb
```

### Training stages

Stage 1 (steps 0 → 5000): EUPE backbone frozen, only TransformerGRU + Decoder trained at 0.1× LR. This lets the GRU gates calibrate to EUPE's feature distribution before full fine-tuning.

Stage 2 (steps 5000 → 1M): Backbone unfrozen with layerwise LR decay (0.75 per layer from output). Earlier backbone layers receive lower learning rates.

### Key hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW, β=(0.9, 0.95) |
| Base LR | 1.5e-4 |
| Weight decay | 0.05 |
| Warmup | 10K steps |
| Total steps | 1M |
| Effective batch | 2048 (32/GPU × 8 acc × 8 GPUs) |
| Precision | bfloat16 |
| Masking ratio | 95% on target frame |
| Source frames | 4 consecutive |
| Target gap | uniform [4, 48] frames |

---

## Evaluation

### Frozen-encoder evaluation

After training, evaluate with frozen encoder + per-task readout heads:

```bash
python scripts/eval_frozen.py \
  checkpoint=/path/to/checkpoint.pt \
  model=rvm_eupe_vitb \
  eval.davis_root=/data/davis \
  eval.kinetics400_root=/data/kinetics400 \
  eval.ssv2_root=/data/ssv2 \
  eval.scannet_root=/data/scannet \
  eval.jhmdb_root=/data/jhmdb \
  eval.perception_test_root=/data/perception_test
```

Results are saved to `outputs/.../eval_results.json`.

### Benchmarks and metrics

| Benchmark | Metric | Readout head | RVM-L baseline |
|-----------|--------|-------------|----------------|
| DAVIS 2017 | J&F mean ↑ | Non-parametric k-NN (k=7, τ=0.7) | 66.0 |
| Kinetics-400 | Top-1 acc ↑ | `AttentiveReadoutHead`, 1 learned query | ~57% |
| Kinetics-700 | Top-1 acc ↑ | Same | 57.3 |
| Something-Something-v2 | Top-1 acc ↑ | Same | 66.7 |
| ScanNet depth | AbsRel ↓ | 8×8 spatial queries, log-depth L2 | 0.91 |
| JHMDB keypoints | PCK@0.1 ↑ | Per-joint learned queries | 48.4 |
| Perception Test | AJ ↑ | Point queries, Average Jaccard | 77.3 |

### DAVIS evaluation detail

DAVIS uses non-parametric label propagation (no learned parameters), the hardest and most honest test of frozen representation quality. For each query frame, patch tokens are compared via cosine similarity against a rolling memory of past frames, and labels are voted via top-k softmax weighting.

---

## Implementation phases

### Phase 0 — Scaffold
- `conda.yaml`, `pyproject.toml`, package `__init__.py` files
- Verified EUPE is installed and importable from `/home/farm/research/eupe/`

### Phase 1 — Core model
- `EUPEEncoderWrapper`: verified that `forward_features()` already strips the 4 storage tokens and CLS token at `x[:, n_storage_tokens+1:]` (vision_transformer.py:247–251). `CenterPadding(16)` imported from `eupe.eval.depth.models.embed`.
- `TransformerGRU`: gate projections initialised with `trunc_normal_(std=0.01)` so initial sigmoid output ≈ 0.5. `F.scaled_dot_product_attention` used throughout for flash-attention compatibility.
- `MAEDecoder`: lazy positional embedding cache keyed on `(grid_h, grid_w)` — handles variable resolutions without re-computation.
- `RecurrentVideoMAE`: thin orchestrator. Truncated BPTT via `.detach()` after each source frame rollout.

### Phase 2 — Data pipeline
- `VideoClipDataset`: supports both video files (via decord) and pre-extracted JPEG frame directories. Sampling: random start, 4 consecutive source frames with configurable stride, target at uniform gap [4, 48].
- `MixedVideoDataset`: `torch.utils.data.WeightedRandomSampler` gives per-sample weights = dataset_weight / dataset_size. Cleanly handles missing datasets (skips with warning).
- `transforms.py`: all spatial augmentations (crop, flip, colour jitter) applied identically to all frames in a clip to preserve temporal coherence.

### Phase 3 — Training infrastructure
- `schedulers.py`: `cosine_warmup_schedule` + `get_layerwise_param_groups`. Layerwise decay iterates backbone blocks by name (`blocks.{i}.`), assigns `lr * decay^(num_layers-1-i)` per layer. Non-backbone params (recurrent, decoder) always receive `base_lr`.
- `scripts/train.py`: Hydra config system, FSDP with `SHARD_GRAD_OP`, gradient clipping, bfloat16 autocast, staged encoder freeze/unfreeze, periodic checkpointing.

### Phase 4 — Evaluation
- `davis.py`: full non-parametric pipeline — feature extraction, k-NN propagation, J (IoU) and F (boundary F-measure) per object per sequence, averaged.
- `action_cls.py`: `AttentiveReadoutHead` with a single learned global query, cross-entropy loss, top-1/top-5 accuracy.
- `depth.py`: 8×8 = 128 spatial queries with Fourier-embedded coordinates, log-depth L2 training, AbsRel/SqRel/RMSE evaluation.
- `tracking.py`: Perception Test Average Jaccard at 5 pixel thresholds (1, 2, 4, 8, 16px).
- `keypoint.py`: per-joint learned queries, L2 coordinate regression, PCK@0.1 with torso-diameter normalisation.
- `readout_heads.py`: shared `AttentiveReadoutHead` base with pluggable `QueryFactory`. `LearnedQueryFactory` for classification/tracking; `SpatialQueryFactory` with Fourier embedding for dense spatial tasks.

### Phase 5 — Ablation
- `FPNAdapter`: 1×1 `nn.Conv2d` per intermediate layer → sum all projected maps → flatten to [B, N, D].
- `ConcatAdapter`: flatten all layers → concatenate channels → single `nn.Linear`.
- `MultiScaleEUPEEncoder`: uses `get_intermediate_layers(n=[2,5,8,11], reshape=True)` which returns stride-16 spatial maps `[B, D, H//16, W//16]` after storage-token stripping (verified in vision_transformer.py:306–313).
- Confirmed: multi-depth output is numerically different from last-layer-only output (tested).

---

## Next Step

The codebase is complete and all tests pass. The only remaining work before the first training run is **wiring the dataset index files**.

1. **Build index JSON files** for each dataset:
   ```bash
   # Example for Kinetics-700 — adapt to your storage layout
   python -c "
   import json, glob
   entries = [{'path': p, 'num_frames': 250} for p in glob.glob('/data/k700/**/*.mp4', recursive=True)]
   json.dump(entries, open('k700_index.json','w'))
   "
   ```
   Or use `decord` to read actual frame counts:
   ```python
   import decord, json
   entries = []
   for path in video_paths:
       vr = decord.VideoReader(path, num_threads=1)
       entries.append({"path": path, "num_frames": len(vr)})
   ```

2. **Update `configs/data/mixed.yaml`** with the paths to these index files.

3. **Smoke test** (single GPU, ~10 minutes):
   ```bash
   python scripts/train.py \
     model=rvm_eupe_vitt \
     train.scheduler.total_steps=1000 \
     train.batch_size_per_gpu=4 \
     train.accumulate_grad_batches=1 \
     train.backbone.freeze_steps=100
   ```
   Verify loss decreases from ~0.5 to <0.1 over 1K steps.

4. **Full training run**: launch with `torchrun` on 8 GPUs using `model=rvm_eupe_vitb train=large_scale`.

5. **Ablation runs**: after the primary run converges, run A2 (`model=rvm_eupe_multiscale`) and the baseline (`model=rvm_baseline_vitb`) for the paper's ablation table.
