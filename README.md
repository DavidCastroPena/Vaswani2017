# Transformer Replication From First Principles

This repository documents a **step-by-step, learning-oriented replication** of the core ideas from  
**“Attention Is All You Need” (Vaswani et al., 2017)**.

The goal is not to chase benchmark scores, but to **reconstruct the Transformer architecture from the ground up**, understand *why* it works, and empirically validate its central claims using clean, inspectable PyTorch code.

This project emphasizes:
- correctness over shortcuts
- interpretability over scale
- scientific replication over cargo-cult implementation

---

## Replication Plan 

The implementation follows a deliberately staged plan, where each step is validated before moving on.

### Step 1 — Build Attention Correctly
**Objective:** Implement scaled dot-product attention from first principles.

- Single-head attention
- Explicit verification of tensor shapes (e.g. `(seq_len, d_model)`)
- Decoder-style causal masking
- Inspection of attention score matrices and outputs

> This step ensures a concrete understanding of how Q, K, V interact and how attention weights are computed.

---

### Step 2 — Add Multi-Head Attention
**Objective:** Generalize attention across multiple representation subspaces.

- Learnable projections for each head
- Parallel attention computation
- Concatenation and output projection
- Sanity check: multi-head attention collapses to single-head behavior when `h = 1`

> This step demonstrates how multi-head attention avoids information loss due to averaging.

---

### Step 3 — Stack Transformer Layers
**Objective:** Reproduce the Transformer block structure faithfully.

- Residual connections
- Layer normalization **after** addition (post-norm, as in the original paper)
- Separation of attention and feed-forward sublayers

> This step focuses on depth, stability, and gradient flow.

---

### Step 4 — Add Positional Encoding
**Objective:** Inject sequence order without recurrence or convolution.

- Sinusoidal positional encodings (original formulation)
- Summation with token embeddings
- Shape and scale consistency checks

> This step highlights how order is represented geometrically rather than sequentially.

---

### Step 5 — Train on Toy Tasks
**Objective:** Empirically test long-range dependency modeling.

- Copy task
- Sequence reversal task
- Controlled experiments to verify that attention handles distant dependencies reliably

> These tasks isolate the core strength of self-attention without dataset noise.

---

### Step 6 — Train on Small Machine Translation
**Objective:** Validate behavior on a realistic NLP task.

- IWSLT14 or a small WMT slice
- Subword tokenization
- Lightweight training setup suitable for single-GPU environments

> Focus is on learning dynamics and qualitative behavior, not absolute BLEU.

---

### Step 7 — Compare Against an RNN Baseline
**Objective:** Test the central hypothesis of the paper.

- Same dataset
- Same parameter budget
- Compare:
  - convergence speed
  - performance as sequence length increases
  - training stability

> **Replication success** is defined as:
> - faster convergence
> - better handling of long-range dependencies
> - stable deep stacking

---

## What This Replication Is Really About

The main scientific contribution of *Attention Is All You Need* is **not BLEU score improvements**.

From Section 4 of the paper (arXiv:1706.03762v7):

> **Self-attention reduces the path length between any two tokens in a sequence to O(1).**

This project therefore focuses on demonstrating:

- **Shorter effective dependency paths**
- **Improved long-range dependency modeling**
- **Highly parallelizable computation**
- **Stable training of deep architectures**

BLEU is treated as a secondary signal, not the goal.

---

## Why This Matters

This replication is designed to show:
- understanding of modern NLP architectures
- comfort with tensor algebra and PyTorch
- ability to translate theory into correct, testable code
- scientific thinking about *what* should be replicated and *why*

The code favors readability and inspection over abstraction, making it suitable for both learning and review.

---

## Status

This repository is actively evolving as part of a structured learning journey into:
- neural networks
- attention mechanisms
- representation learning
- NLP system design
