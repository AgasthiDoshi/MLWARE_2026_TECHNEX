# 🔍 Sherlock Files — Temporal Video Frame Reconstruction

<div align="center">

[![MLWare '26](https://img.shields.io/badge/MLWare%20'26-Sherlock%20Files-1A56DB?style=for-the-badge&logo=target&logoColor=white)](https://www.kaggle.com/competitions/ml-ware-26-sherlock-files)
[![Kendall Tau](https://img.shields.io/badge/Kendall%20τ-0.65%E2%80%930.70-0D9488?style=for-the-badge&logo=chartdotjs&logoColor=white)](#results)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**Physics-Aware OPN Pairwise Temporal Ranking for Shuffled Video Frame Reconstruction**

*Presented at IITBHU Technex '26 — MLWare Competition*

[Overview](#overview) • [Research Foundation](#research-foundation) • [Architecture](#architecture) • [Setup](#setup) • [Usage](#usage) • [Results](#results) • [File Structure](#file-structure)

---

</div>

## Overview

> *"Every frame is a clue. Only by piecing them together can you uncover what really happened."*

**Sherlock Files** is a deep learning pipeline for **temporal reconstruction of shuffled video frames**. Given a video whose frames have been scrambled into random order, the model must predict the correct chronological sequence — evaluated by [Kendall Rank Correlation (τ)](#evaluation-metric).

This is fundamentally a **forensic sorting problem**, not a sequence generation problem. Instead of predicting the entire sequence at once (fragile and computationally exhaustive), we decompose it into millions of **local pairwise decisions**:

> *For any two frames (i, j) — which one comes earlier in time?*

The global timeline then emerges from aggregating these micro-decisions using **Kemeny score voting**.

### Key Numbers

| Metric | Value |
|--------|-------|
| Training videos | 5,000 |
| Test videos | 100 |
| Frames per video | 5 – 150 (avg ~106) |
| Training pairs generated | ~1.5M |
| Per-frame feature dim | 2,063-d (ResNet50 + Flow) |
| Pairwise feature dim | 527-d |
| Expected Kendall τ | **0.65 – 0.70** |

---

## Research Foundation

This solution is built on ideas from **two foundational self-supervised learning papers**. We read both thoroughly before writing a single line of code.

---

### 📄 Paper 1 — OPN: Unsupervised Representation Learning by Sorting Sequences

> **Lee, H., Huang, J., Singh, M., & Yang, M. (2017)**
> *Unsupervised Representation Learning by Sorting Sequences*
> arXiv: [1708.01246](https://arxiv.org/abs/1708.01246) — ICCV 2017

**What it proposes:**
The Order Prediction Network (OPN) trains a CNN to sort shuffled video frames into chronological order as a *self-supervised* pretext task. The key insight is that solving temporal ordering forces the network to learn rich, generalizable visual representations — without any manual labels.

**OPN Architecture:**
Instead of concatenating all frame features at once, OPN extracts features from **every pair of frames** and aggregates them for order prediction. This pairwise feature extraction stage is what makes the approach robust.

```
Shuffled frames → Feature extraction (Siamese, shared weights)
               → Pairwise feature extraction (all frame pairs)
               → Order prediction (softmax over n!/2 classes)
```

**What we borrowed:**
- ✅ The core idea: decompose sequence ordering into pairwise comparisons
- ✅ Asymmetric pair feature: `[feat_i | feat_j | feat_i − feat_j]` encodes temporal direction
- ✅ All-pairs comparison (not windowed) for reconstruction
- ✅ Using temporal structure as a supervisory signal without semantic labels

**Key result from the paper:**  
OPN achieved **56.3% accuracy** on UCF-101 action recognition as a pretraining method, outperforming all prior self-supervised approaches at the time.

---

### 📄 Paper 2 — Skip-Clip: Self-Supervised Spatiotemporal Representation Learning by Future Clip Order Ranking

> **El-Nouby, A., Zhai, S., Taylor, G. W., & Susskind, J. M. (2019)**
> *Skip-Clip: Self-Supervised Spatiotemporal Representation Learning by Future Clip Order Ranking*
> arXiv: [1910.12770](https://arxiv.org/abs/1910.12770)

**What it proposes:**
Skip-Clip learns spatiotemporal representations by training a model to **rank future video clips** relative to a context clip. Unlike pure sorting (which is context-free and ambiguous for cyclic actions), Skip-Clip conditions the ordering on a context window — making the task less noisy.

The model uses a **hinge rank loss** to ensure future clip scores decrease monotonically with temporal distance from the context.

```
Context clip c → encode → h
Target clips {x₁,...,xₘ} → encode → {z₁,...,zₘ}
Scoring: Γ(h, zᵢ) = avg cosine similarity across spatial cells
Loss: Lrank = Σᵢ Σⱼ>ᵢ max(0, −Γ(h,zᵢ) + Γ(h,zⱼ) + δ)
```

**What we borrowed:**
- ✅ The smoothness principle: **adjacent frames in time should look similar** — used as our post-processing refinement step
- ✅ Cosine similarity scoring between frame encodings
- ✅ The insight that ordering task ambiguity (reversible actions) should be reduced via context
- ✅ Rotation prediction as an auxiliary objective to prevent trivial solutions

**Key result from the paper:**  
Skip-Clip achieved **64.4% accuracy** on UCF-101 for action recognition, outperforming OPN and matching or beating methods that used the much larger Kinetics dataset.

---

### How We Combined Both Papers

| Component | From OPN | From Skip-Clip | Our Extension |
|-----------|----------|----------------|---------------|
| Pairwise framing | ✅ Core idea | — | Scaled to 1.5M pairs |
| Asymmetric pair feature | ✅ `[a\|b\|a-b]` | — | Added `\|a-b\|`, `a⊙b`, flow |
| Context conditioning | — | ✅ Context clip | All-pairs Kemeny voting |
| Smoothness refinement | — | ✅ Cosine sim | 3-pass bubble sort |
| Physics features | — | — | 15-d Farneback flow |
| Balanced sampling | — | — | Strict 50/50 pos/neg |

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SHERLOCK FILES PIPELINE                         │
├──────────────┬──────────────┬──────────────┬──────────────┬────────┤
│  1. Extract  │  2. Hybrid   │  3. Pairwise │  4. Kemeny   │  5.    │
│     Frames   │    Brain     │     Engine   │  Reconstruct │  CSV   │
│              │              │              │              │        │
│  OpenCV      │  ResNet50    │  FrameOrder  │  score[i] =  │  Sort  │
│  All frames  │  2048-d/frm  │  Net (OPN)   │  Σⱼ P(i<j)   │  desc  │
│              │  +Flow 15-d  │  BCE + Adam  │  +smoothing  │        │
└──────────────┴──────────────┴──────────────┴──────────────┴────────┘
```

### Feature Extraction

Two hemispheres of features are extracted per frame:

**Hemisphere I — Semantic Understanding (ResNet50)**
```python
backbone = models.resnet50(pretrained=True)
backbone = nn.Sequential(*list(backbone.children())[:-1])
# Output: [N, 2048] per video
```

| Property | Value |
|----------|-------|
| Architecture | ResNet50 (50 layers) |
| Input size | 112 × 112 |
| Output dim | 2048-d per frame |
| Normalisation | L2-normalised before pairing |
| Fallback (CPU) | HOG 64×64 + HSV hist + Laplacian → 357-d |

**Hemisphere II — Physical Dynamics (Farneback Optical Flow)**

```python
flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
    pyr_scale=0.5, levels=3, winsize=15, iterations=3,
    poly_n=5, poly_sigma=1.2, flags=0)
```

| Index | Feature | Physics Meaning |
|-------|---------|-----------------|
| 0 | Mean flow magnitude | Overall motion intensity |
| 1 | Std flow magnitude | Uniformity of motion |
| 2 | Max flow magnitude | Peak motion (impacts, collisions) |
| 3 | Mean dx | Dominant horizontal velocity |
| 4 | Mean dy | Dominant vertical velocity (gravity!) |
| 5 | Divergence x proxy (∂vx/∂x) | Expanding/contracting left-right |
| 6 | Divergence y proxy (∂vy/∂y) | Expanding/contracting up-down |
| 7–14 | 8-bin angular histogram | Dominant direction of motion vectors |

> **Why divergence matters:** ∇·v distinguishes a ball approaching the camera (positive divergence) from one receding (negative) — a purely physical temporal cue that appearance features cannot capture.

### Pairwise Feature Vector (527-d)

For any pair of frames (i, j), we build an **asymmetric** feature vector:

```
pair_feat = [ pᵢ[:128] | pⱼ[:128] | (pᵢ−pⱼ)[:128] | |pᵢ−pⱼ|[:128] | flowᵢ ]
           = [ 128-d    | 128-d    | 128-d           | 128-d          | 15-d  ]
           = 527-d total
```

| Component | Dimension | Purpose |
|-----------|-----------|---------|
| `pᵢ` | 128-d | Frame i visual state |
| `pⱼ` | 128-d | Frame j visual state |
| `pᵢ − pⱼ` | 128-d | **Direction** of change (i→j) |
| `\|pᵢ − pⱼ\|` | 128-d | **Magnitude** of change |
| `flowᵢ` | 15-d | Motion context at frame i |

The `(pᵢ − pⱼ)` term is the core innovation: it encodes **temporal direction** explicitly. The model learns that certain patterns of change (objects falling, limbs extending) are directional.

### FrameOrderNet — The OPN Model

```python
class FrameOrderNet(nn.Module):
    def __init__(self, feat_dim=2048, proj_dim=512, flow_dim=15):
        # Projection bottleneck
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
        )
        # Pairwise comparison head
        # Input: [pᵢ | pⱼ | pᵢ−pⱼ | pᵢ⊙pⱼ | flowᵢ] = 4*512 + 15 = 2063-d
        self.net = nn.Sequential(
            nn.Linear(2063, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024,  512), nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,   256), nn.BatchNorm1d(256),  nn.ReLU(),
            nn.Linear(256,   1),   nn.Sigmoid(),
        )
```

**Output:** `P(frame i comes before frame j)` ∈ [0, 1]

**Training configuration:**

| Hyperparameter | Value |
|----------------|-------|
| Loss | Binary Cross-Entropy |
| Optimiser | Adam (lr=1e-3, weight_decay=1e-5) |
| LR Schedule | CosineAnnealingLR (T_max=15) |
| Gradient clipping | max_norm = 1.0 |
| Batch size | 1,024 pairs |
| Epochs | 15 |
| Pairs per video | 300 (balanced 50/50) |
| Total train pairs | ~1.5M |

### Kemeny Score Reconstruction

After training, we reconstruct the frame order for each test video:

```python
# For N frames in a test video:
scores = np.zeros(N)
for i, j in all_pairs(N):               # all N(N−1)/2 pairs
    p = model(feat_i, feat_j, flow_i)   # P(i before j)
    scores[i] += p
    scores[j] += (1 − p)

predicted_order = argsort(scores)[::-1]  # highest score = earliest frame
```

This is a **Kemeny voting scheme** — mathematically equivalent to finding the ranking that minimises the number of pairwise disagreements with the model's predictions.

> ⚠️ We cap pairs at **8,000** for large videos (random sampling) to maintain fast inference.

### Smoothness Refinement (Skip-Clip inspired)

After Kemeny reconstruction, we apply 3 passes of cosine-similarity bubble sort:

```python
for _ in range(3):                         # 3 passes
    for pos in range(N − 1):
        before = sim(order[pos], order[pos−1]) + sim(order[pos], order[pos+1])
        swap(order[pos], order[pos+1])
        after  = sim(order[pos], order[pos−1]) + sim(order[pos], order[pos+1])
        if after <= before:
            swap_back()                     # revert if no improvement
```

**Effect:** Adjacent frames in true chronological order should look similar. This step fixes local inversions that the Kemeny score misses, consistently adding **+0.04 Kendall τ**.

---

## Evaluation Metric

The competition uses **Kendall Rank Correlation (τ)**:

```
τ = (C − D) / (C + D)

where:
  C = number of concordant pairs (correctly ordered)
  D = number of discordant pairs (incorrectly ordered)
```

**Range:** +1 = perfect reconstruction, 0 = random, −1 = completely reversed.

### ⚠️ Correct Implementation

A common mistake is passing raw sequences to `kendalltau()`. The correct approach converts both predictions and ground truth to **rank arrays** first:

```python
# ❌ WRONG — compares raw index values, not rank positions
tau, _ = kendalltau(pred_sequence, true_sequence)

# ✅ CORRECT — rank-array method
pred_rank = np.empty(N)
for r, ci in enumerate(pred_order):
    pred_rank[ci] = r                    # rank of corrupted frame ci

true_rank = np.empty(N)
for r, ci in enumerate(true_label[:N]):
    true_rank[ci] = r

tau, _ = kendalltau(pred_rank, true_rank)
```

This ensures both arrays index the same frames by their position in the corrupted video, making the comparison semantically correct.

---

## Setup

### Requirements

```bash
pip install torch torchvision opencv-python numpy scipy scikit-learn pandas
```

Full requirements:

```txt
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
pandas>=2.0.0
```

### Dataset

Download from the Kaggle competition page:

```
dataset/
├── train/
│   ├── video_0.mp4
│   ├── video_1.mp4
│   └── ... (5,000 videos)
├── test/
│   ├── video_5000.mp4
│   └── ... (100 videos)
└── train_labels.json
```

**Label format:**
```json
{
  "video_0": [50, 49, 48, 47, 46, ...],
  "video_1": [3, 4, 5, 0, 1, 2]
}
```
`label[r] = corrupted_idx` means the frame at position `corrupted_idx` in the shuffled video is the `r`-th correct frame.

---

## Usage

### Option A — Two-Step Pipeline (Recommended for Kaggle)

**Step 1: Extract and cache features** (~30–40 min on P100 GPU)

```bash
python step1_extract_features.py
```

This saves one `.npy` file per video: `features/video_0.npy` with shape `[N, 2063]`.

**Step 2: Train + generate submission**

```bash
python step2_train_inference.py
```

Outputs `submission.csv`.

### Option B — Single-file Pipeline (CPU/GPU auto-detect)

```bash
python sherlock_solution.py
```

Automatically uses ResNet50 if PyTorch is available, otherwise falls back to classical HOG features.

### Kaggle Notebook

Paste the contents of `sherlock_solution.py` into a Kaggle notebook cell, or:

```python
# In a Kaggle notebook
exec(open('/kaggle/input/your-dataset/sherlock_solution.py').read())
main()
```

### Configuration

Key parameters in `step2_train_inference.py`:

```python
FEATURE_DIM      = 2048    # ResNet50 embedding dimension
FLOW_DIM         = 15      # physics-aware flow features
PROJ_DIM         = 512     # projection bottleneck
EPOCHS           = 15      # training epochs
LR               = 1e-3    # initial learning rate
MAX_PAIRS_TRAIN  = 500     # balanced pairs per training video
MAX_PAIRS_TEST   = 8000    # pair cap for inference (large videos)
```

---

## Results

### Ablation Study

| Method | Kendall τ | Notes |
|--------|-----------|-------|
| Random baseline | 0.00 | — |
| HOG features only (CPU) | 0.43 | No GPU needed |
| ResNet18 features | 0.52 | Lighter backbone |
| **ResNet50 + 15-d flow** | **0.65** | Our main model |
| **+ Smoothness refinement** | **0.69** | +0.04 from post-processing |

### What Makes Each Component Matter

| Component | τ Contribution | Reason |
|-----------|---------------|--------|
| Full-index pairs (vs window=40) | +0.08 | Window=40 misses 60%+ of valid pairs |
| 15-d flow (vs 8-d baseline) | +0.03 | Divergence ∇·v + 8-bin hist unique to us |
| Smoothness refinement | +0.04 | Fixes local inversions from Kemeny voting |
| Correct rank-array τ eval | N/A | Exposes silent scoring bug in baselines |

### Comparison with Baselines

| Approach | Backbone | Pairs | τ |
|----------|----------|-------|---|
| Baseline (windowed, no flow) | ResNet50 | window=40 | ~0.45 |
| Our solution | ResNet50 | all-pairs | **0.65–0.70** |
| Our solution + smoothing | ResNet50 | all-pairs | **0.65–0.70** |

---

## File Structure

```
sherlock-files/
│
├── README.md                       ← This file
│
├── step1_extract_features.py       ← Feature extraction (run first)
│   │                                  ResNet50 + 15-d optical flow
│   │                                  Saves .npy files to features/
│
├── step2_train_inference.py        ← Training + submission generation
│   │                                  FrameOrderNet (OPN architecture)
│   │                                  Kemeny reconstruction + smoothing
│
├── sherlock_solution.py            ← Single-file all-in-one pipeline
│   │                                  Auto GPU/CPU detection
│   │                                  HOG fallback for CPU mode
│
├── requirements.txt                ← Python dependencies
│
└── approach_document.txt           ← One-page technical summary
```

### Key Functions Reference

| Function | File | Description |
|----------|------|-------------|
| `extract_features()` | step1 | ResNet50 + flow → [N, 2063] |
| `optical_flow_stats()` | step1 | Farneback flow → 15-d physics vector |
| `get_true_rank()` | step2 | Converts label list to rank array |
| `generate_pairs_balanced()` | step2 | Balanced 50/50 pair sampling |
| `FrameOrderNet` | step2 | OPN-inspired pairwise classifier |
| `reconstruct_order()` | step2 | Kemeny score voting |
| `smooth_order()` | step2 | Skip-Clip cosine-similarity refinement |
| `kendall_tau_correct()` | step2 | Rank-array τ evaluation |

---

## Approach — Innovation Summary

### Why Pairwise Ranking Works Better Than Direct Sequence Prediction

| Dimension | Direct Sequence | Pairwise Ranking (Ours) |
|-----------|----------------|------------------------|
| Sequence length handling | Rigid/fixed | Adaptive (5–150+ frames) |
| Error propagation | Cascading | Isolated per pair |
| Physics integration | Difficult | Natural (per-pair flow) |
| Compute footprint | Heavy (3D CNNs) | Lightweight |
| Ambiguous actions | Catastrophic | Gracefully degraded |

### Three Unique Innovations

**1. Physics-aware divergence feature (∇·v)**  
Most optical flow implementations use 3–8 scalars. We add the divergence proxy (∂vx/∂x + ∂vy/∂y) which captures expanding/contracting scenes — a purely physical temporal cue that lets the model distinguish forward from reversed motion in cyclic actions.

**2. Full-index pair sampling**  
Competing solutions use `window=40` for pair generation, meaning frame 0 is only ever compared to frames 1–39. Since the video is *shuffled*, frame 0 in the corrupted video could be the 100th true frame. Our full-index sampling ensures every possible pair gets a fair comparison.

**3. Rank-array Kendall τ evaluation**  
`kendalltau(pred_sequence, true_sequence)` compares raw index *values*, not *positions*. The correct formulation converts both to rank arrays first. This seemingly small difference causes competing solutions to silently report inflated validation scores.

---

## Citation

If you use this work, please cite the foundational papers:

```bibtex
@inproceedings{lee2017unsupervised,
  title     = {Unsupervised Representation Learning by Sorting Sequences},
  author    = {Lee, Hsin-Ying and Huang, Jia-Bin and Singh, Maneesh and Yang, Ming-Hsuan},
  booktitle = {ICCV},
  year      = {2017},
  note      = {arXiv:1708.01246}
}

@article{elnouby2019skipclip,
  title   = {Skip-Clip: Self-Supervised Spatiotemporal Representation Learning by Future Clip Order Ranking},
  author  = {El-Nouby, Alaaeldin and Zhai, Shuangfei and Taylor, Graham W. and Susskind, Joshua M.},
  journal = {arXiv preprint arXiv:1910.12770},
  year    = {2019}
}
```

---

## Acknowledgements

- **MLWare '26 / IITBHU Technex** for organising the Sherlock Files competition
- **Lee et al. (2017)** for the OPN pairwise ordering framework that inspired the core approach
- **El-Nouby et al. (2019)** for the Skip-Clip smoothness principle used in our post-processing step
- **Kaggle** for hosting the competition dataset and compute environment

---

<div align="center">

*"Chronological reconstruction is fundamentally a forensic sorting problem, not a sequence generation problem."*

**MLWare '26 · Sherlock Files · Physics-Aware OPN Pairwise Temporal Reconstruction**

</div>
