# TrainingHelp.md

# NSAI-K1 Training Guide

This document explains how to train and improve NSAI-K1 locally.

---

# Requirements

Install all required dependencies before training.

```bash
pip install -r requirements.txt
```

Recommended Python version:

```text
Python 3.10+
```

---

# Dataset Format

NSAI-K1 uses `.jsonl` datasets.

Dataset location:

```bash
data/dataset.jsonl
```

Each line must contain one JSON object.

Example:

```json
{"prompt":"Hello","response":"Hi there!"}
{"prompt":"What is Python?","response":"Python is a programming language."}
```

Rules:

* One JSON object per line
* Use valid JSON formatting
* Avoid duplicate prompts
* Keep responses clean
* Do not leave empty lines

---

# Current Dataset Status

| Version | Approx Dataset Size  |
| ------- | -------------------- |
| NSAI-K1 | ~3K lines            |
| NSAI-K2 | ~30K lines (planned) |

NSAI-K1 is currently a lightweight research project with a small dataset focused on experimentation and local Transformer development.

Future versions will include:

* Larger datasets
* Better contextual understanding
* Improved memory handling
* Cleaner responses
* Faster inference

---

# Starting Training

Run training with:

```bash
python train.py
```

Default training process includes:

* Tokenizer training
* Dataset chunking
* Transformer training
* Automatic checkpoint saving

---

# Training Parameters

Main configurable settings:

| Parameter      | Default |
| -------------- | ------- |
| Context Size   | 128     |
| Embedding Size | 192     |
| Layers         | 4       |
| Heads          | 4       |
| Epochs         | 20      |
| Batch Size     | 16      |
| Learning Rate  | 3e-4    |

---

# Custom Training Example

Train with custom epochs:

```bash
python train.py --epochs 40
```

Train with larger batch size:

```bash
python train.py --batch_size 32
```

Train using CUDA:

```bash
python train.py --device cuda
```

---

# Output Files

After training, generated files are stored inside:

```bash
checkpoints/
```

Files:

* `mini_transformer.pt`
* `tokenizer.json`
* `config.json`

These files are required for inference and deployment.

---

# Running the AI

## Terminal Version

```bash
python AI.py
```

## Streamlit Web Interface

```bash
streamlit run app.py
```

---

# Memory System

NSAI-K1 automatically stores useful conversations into:

```bash
data/dataset.jsonl
```

The memory system:

* Prevents duplicates
* Stores useful responses
* Performs similarity matching
* Improves lightweight local recall

---

# Improving Model Quality

Recommended improvements:

## Increase Dataset Size

Larger datasets significantly improve:

* Response quality
* Context handling
* Memory
* Reasoning

---

## Improve Dataset Quality

Better training data produces better outputs.

Recommended topics:

* Programming
* Mathematics
* AI Discussions
* Cyber Security
* Educational Q&A
* Science
* Linux Commands

Avoid:

* Corrupted text
* Spam
* Duplicate content
* Random internet dumps

---

## Increase Model Size

Inside `train.py`:

```python
d_model = 256
n_layers = 6
```

Warning:
Larger models require more RAM and GPU resources.

---

# GPU Support

NSAI-K1 supports CUDA acceleration.

Check CUDA availability:

```python
import torch
print(torch.cuda.is_available())
```

Run with GPU:

```bash
python train.py --device cuda
```

---

# Common Errors

## Missing Modules

Install dependencies again:

```bash
pip install -r requirements.txt
```

---

## CUDA Error

Fallback to CPU:

```bash
python train.py --device cpu
```

---

## Dataset Error

Check:

* Dataset path
* JSON formatting
* Empty lines
* Corrupted entries

---

# Recommended Hardware

Minimum:

* 8GB RAM
* Intel i3 / Ryzen 3
* Python 3.10

Recommended:

* 16GB RAM
* NVIDIA GPU
* SSD Storage

---

# Upcoming Improvements

Planned for NSAI-K2:

* ~30K dataset
* Better Transformer architecture
* Improved tokenizer
* Faster response generation
* Better reasoning
* Advanced memory optimization
* BEP Tech integration
* More stable training pipeline

---

# Author

Navtej Singh Saggar

NSAI-K1 is a lightweight Transformer research project focused on local AI experimentation and development.
