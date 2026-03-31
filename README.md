# PitchVision: Automated Football Scouting Agent

An end-to-end computer vision system that processes full-match tactical camera footage
and produces per-player scouting reports — combining on-ball event detection with
off-ball movement intelligence.

**Every neural network in this project is implemented from scratch using only `torch.nn`.**
No pretrained detection libraries. No tracking libraries. No high-level ML frameworks.
Five architectures, one codebase, pure PyTorch.

---

## Pure PyTorch Philosophy

This project exists at the intersection of applied CV and deep learning education.
Instead of importing YOLOv8 from ultralytics or DeepSORT from a pip package,
every architecture is built from first principles:

| Architecture | What It Does | Key Paper |
|---|---|---|
| **ResNet-18 backbone** | Feature extraction from video frames | He et al. 2015 |
| **YOLO detection head** | Single-shot player + ball detection | Redmon et al. 2016 |
| **DeepSORT tracker** | Multi-object tracking with persistent IDs | Wojke et al. 2017 |
| **Tactical Transformer** | Off-ball intent prediction from tracking tokens | Vaswani et al. 2017 |
| **Attention Rollout** | Interpretability via attention flow composition | Abnar & Zuidema 2020 |

Supporting components also built from scratch:
- **Kalman Filter** — 8D state prediction/update for motion modeling
- **ReID Network** — CNN producing 128-dim appearance embeddings for re-identification
- **Non-Maximum Suppression** — IoU computation + greedy suppression
- **Fourier Positional Encoding** — Random Fourier features for 2D continuous coordinates
- **Multi-head decoder** — 4 independent MLP prediction heads

The only external ML dependency is `torchvision` (for loading pretrained ImageNet weights
into our from-scratch ResNet-18). Everything else: `torch`, `numpy`, `scipy` (just
`linear_sum_assignment`), `opencv` (just I/O), and `ruptures` (changepoint detection).

---

## System Capabilities

### On-Ball Analysis

Derived from tracking data using geometric heuristics — no separate ML model needed.

| Feature | Method | Output |
|---|---|---|
| **Pass detection** | Ball transitions between players (proximity + velocity) | Passer, receiver, success/fail, distance |
| **Shot detection** | Ball moves toward goal at high velocity from player vicinity | Shooter, on-target, distance to goal |
| **Dribble/carry detection** | Ball remains near player while player displaces > 2m | Carrier, distance, duration |
| **Reception** | Ball arrives near player (inverse of pass) | Receiver, first-touch quality proxy |
| **Pressing trigger** | Rapid closing of distance to ball carrier | Presser, intensity (speed of closure) |

These events are accumulated into a **match event log** — a time-stamped record of
every detected action, attributable to specific players.

### Off-Ball Analysis

Predicted by the from-scratch tactical transformer operating on player tracking tokens.

| Feature | Model Output | Scouting Value |
|---|---|---|
| **Movement intent** | 5 classes: support, press, create-space, hold, unknown | Tactical role profiling |
| **Run direction** | 8 compass directions | Attacking pattern analysis |
| **Urgency score** | 0–1 continuous | Intensity and work rate |
| **Predicted next position** | Δ(x,y) at 1 second | Anticipation and positioning quality |
| **Causal attention** | Which opponent triggered this run | Reactive vs. proactive movement |

### Scouting Queries

The system answers natural-language scouting questions over full matches:

```
"How many passes did #5 make?"               → event log count
"Who pressed the most in the second half?"    → intent aggregation + time filter
"Show me all times #8 lost possession"        → event sequence pattern
"Compare the work rate of both center-mids"   → urgency + distance + sprint stats
"Which winger creates the most space?"        → intent=space count by player
```

Simple queries hit a structured API. Complex queries route through an LLM
with the match event log as context.

---

## Technical Architecture

```
YouTube Tactical Cam Video
         │
    ┌────┴──────────┐
    │  ResNet-18     │──► Multi-scale feature maps (feat16, feat32)
    │  (from scratch)│
    └────┬──────────┘
         │
    ┌────┴──────────┐
    │  YOLO Head     │──► Player detections + Ball detections
    │  (from scratch)│     (grid decode → NMS)
    └────┬──────────┘
         │
    ┌────┴──────────┐
    │  DeepSORT      │──► Player tracks (consistent IDs across frames)
    │  (from scratch)│     Kalman filter + ReID + Hungarian matching
    └────┬──────────┘
         │
    ┌────┴─────────┐
    │  Homography   │──► Pixel coords → Field coords (meters)
    │  (cv2 only)   │
    └────┬─────────┘
         │
         ├──────────────────────────────────┐
         │                                  │
    ┌────┴──────────┐               ┌───────┴──────────────┐
    │ Event Detector │               │ Tactical Transformer  │
    │  (heuristic)   │               │   (from scratch)      │
    │                │               │                       │
    │ • pass         │               │ • Fourier pos enc     │
    │ • shot         │               │ • player tokenizer    │
    │ • dribble      │               │ • 6-layer encoder     │
    │ • reception    │               │ • multi-head decoder  │
    │ • press trigger│               │ • attention rollout   │
    └────┬──────────┘               └───────┬──────────────┘
         │                                  │
         └──────────┬───────────────────────┘
                    │
            ┌───────┴───────┐
            │Match Accumulator│
            │                 │
            │ Per-player stats│
            │ Event timeline  │
            │ Scouting profile│
            └───────┬───────┘
                    │
            ┌───────┴───────┐
            │  Query Layer   │
            │                │
            │ Structured API │
            │ + LLM agent    │
            └───────────────┘
```

---

## From-Scratch Architectures

### 1. ResNet-18 Backbone (`pitchvision/detection/backbone.py`)

- **BasicBlock**: 3×3 conv → BN → ReLU → 3×3 conv → BN → skip connection
- **Shortcut projections**: 1×1 conv when dimensions change
- **Multi-scale output**: feat16 (256 channels) and feat32 (512 channels)
- ImageNet weights loaded after verifying our architecture matches torchvision's

**Paper**: "Deep Residual Learning for Image Recognition" (He et al., 2015)

### 2. YOLO Detection Head (`pitchvision/detection/head.py`)

- Grid-based single-shot detection from backbone features
- Anchor-based decoding: sigmoid centers + exp sizes + grid offsets
- Per-class confidence via objectness × class probability
- Greedy NMS with IoU thresholding

**Paper**: "You Only Look Once" (Redmon et al., 2016)

### 3. DeepSORT Tracker (`pitchvision/tracking/`)

- **Kalman Filter**: 8D state (cx, cy, aspect, height + velocities), constant velocity model
- **ReID Network**: Small CNN → 128-dim L2-normalized appearance embeddings
- **Cascade Matching**: Cosine distance on appearances → IoU fallback → Hungarian algorithm
- **Track Lifecycle**: Tentative → Confirmed (after n_init hits) → Deleted (after max_age misses)

**Paper**: "Simple Online and Realtime Tracking with a Deep Association Metric" (Wojke et al., 2017)

### 4. Tactical Transformer (`pitchvision/models/`)

~2M parameter temporal transformer on structured player tokens.

- **Input**: 5-frame window × (22 players + 1 ball) = 115 tokens
- **Token**: Fourier position (43d) + kinematics (43d) + role embedding (42d) → 128d
- **Encoder**: 6 layers, 8 heads, pre-norm, GELU FFN (d_model=128, ffn=512)
- **Decoder**: 4 independent MLP heads (direction, intent, urgency, next_pos)
- **Rollout**: Attention flow composition for ball→player causal influence

**Papers**: Vaswani 2017 (attention), Dosovitskiy 2020 (pre-norm), Tancik 2020 (Fourier features)

---

## Data Sources

| Source | Cost | What It Provides |
|---|---|---|
| YouTube tactical cam videos | Free | Full-match wide-angle footage |
| SoccerNet-v3 tracking data | Free | 10Hz (x,y) positions for 500+ matches (for training) |
| StatsBomb open data | Free | Event-level match data for validation |

**Training data**: SoccerNet tracking coordinates with auto-generated weak labels
(intent derived from trajectory geometry — no manual annotation needed).

**Inference target**: YouTube tactical cam videos processed through the full pipeline.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | GTX 1660 (6GB) | RTX 4090 (24GB) |
| RAM | 16GB | 32GB |
| Storage | 20GB (SoccerNet subset) | 50GB (full dataset) |

Training: ~2-4 hours on RTX 4090 for supervised fine-tuning.
Inference: < 100ms per frame on CPU, < 20ms on GPU.

---

## Project Structure

```
pitch_vision/
├── configs/
│   ├── base.yaml              # Model + data hyperparameters
│   └── finetune.yaml          # Training config
│
├── pitchvision/
│   ├── detection/             # Week 1: from-scratch detector
│   │   ├── backbone.py        #   ResNet-18 (BasicBlock + skip connections)
│   │   ├── head.py            #   YOLO detection head (grid decode)
│   │   ├── nms.py             #   IoU + greedy NMS
│   │   └── detector.py        #   Full detector assembly
│   │
│   ├── tracking/              # Week 2: from-scratch tracker
│   │   ├── kalman.py          #   Kalman filter (8D state, predict/update)
│   │   ├── reid.py            #   ReID CNN (128-dim embeddings)
│   │   ├── track.py           #   Track lifecycle management
│   │   ├── matching.py        #   Cost matrices + Hungarian matching
│   │   └── tracker.py         #   DeepSORT assembly (cascade matching)
│   │
│   ├── models/                # Week 3: from-scratch transformer
│   │   ├── attention.py       #   Multi-head self-attention
│   │   ├── encodings.py       #   Fourier positional + frame embeddings
│   │   ├── token.py           #   Player token construction
│   │   ├── decoder.py         #   4-head intent decoder
│   │   ├── rollout.py         #   Attention rollout (interpretability)
│   │   └── pitchvision.py     #   Full model assembly
│   │
│   ├── video/                 # Homography + pipeline
│   │   ├── homography.py      #   Pixel ↔ field coordinate mapping
│   │   └── pipeline.py        #   Video → tracking JSON
│   │
│   ├── data/                  # Data pipeline (Week 2-3)
│   ├── training/              # Training pipeline (Week 3-4)
│   ├── inference/             # Match processor + accumulator (Week 4)
│   └── utils/config.py        # YAML config loader
│
├── server/                    # FastAPI scouting API
├── scripts/                   # CLI entry points
├── tests/                     # Unit + integration tests
├── pyproject.toml
└── Makefile
```

---

## Project Status

> 4-week sprint starting March 18, 2026.

- [x] Week 1 — Detection from scratch (ResNet-18 backbone + YOLO head + NMS)
- [ ] Week 2 — Tracking from scratch (Kalman filter + ReID + DeepSORT)
- [ ] Week 3 — Transformer from scratch (attention + encoder + decoder + rollout)
- [ ] Week 4 — Integration, training, inference pipeline, scouting API

### Papers Reading List

| Week | Paper | Focus |
|---|---|---|
| 1 | "Deep Residual Learning" (He 2015) | Skip connections, BasicBlock |
| 1 | "You Only Look Once" (Redmon 2016) | Grid-based detection, anchor decoding |
| 2 | "Simple Online and Realtime Tracking" (Wojke 2017) | Cascade matching, ReID |
| 3 | "Attention Is All You Need" (Vaswani 2017) | Scaled dot-product, multi-head |
| 3 | "An Image is Worth 16x16 Words" (Dosovitskiy 2020) | Pre-norm transformer |
| 3 | "Fourier Features" (Tancik 2020) | Random Fourier features for 2D coords |
| 3 | "Attention Rollout" (Abnar & Zuidema 2020) | Attention flow composition |
| 4 | "Focal Loss" (Lin 2017) | Class imbalance handling |
