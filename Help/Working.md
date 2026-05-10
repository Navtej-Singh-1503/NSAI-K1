# Working.md

# How NSAI-K1 Works

This document explains the internal working flow of NSAI-K1.

---

# Overview

NSAI-K1 is a lightweight Transformer-based AI assistant built using PyTorch.

The project combines:

* Transformer Language Modeling
* Local memory storage
* Web research
* Wikipedia fallback
* Lightweight response generation

NSAI-K1 is designed as a local experimental AI system focused on simplicity and research.

---

# Core Workflow

The AI follows this response pipeline:

```text id="x2v08n"
User Input
   ↓
Memory Search
   ↓
Math Solver
   ↓
Web Search
   ↓
Wikipedia Fallback
   ↓
Response Output
   ↓
Memory Save
```

---

# 1. User Input

The user sends a message through:

* Terminal interface
* Streamlit web interface

Example:

```text id="ikv0ww"
You: Python bubble sort code
```

The input is cleaned using:

```python id="pb7oq6"
user_input = input().strip()
```

---

# 2. Memory Search

NSAI-K1 first checks stored memory inside:

```bash id="yy0xdu"
data/dataset.jsonl
```

The system compares similarity between:

* Current user prompt
* Previously saved prompts

Using:

```python id="m8zv7w"
SequenceMatcher
```

If similarity is high enough:

```python id="f2gryn"
similarity > 0.92
```

The AI returns the saved response instantly.

Purpose:

* Faster replies
* Lightweight memory recall
* Duplicate prevention

---

# 3. Math Engine

If memory does not contain an answer, NSAI-K1 checks for mathematical expressions.

Example:

```text id="9d2gl4"
25 + 17 * 3
```

The system uses:

* Regex pattern matching
* Python evaluation

Inside:

```python id="l7ppqh"
solve_math()
```

Example output:

```text id="tfv6bq"
Calculation Result: 76
```

---

# 4. Web Research System

If no local answer is found, NSAI-K1 performs lightweight web research.

Uses:

* DuckDuckGo Search (`ddgs`)
* BeautifulSoup scraping

Search example:

```python id="9bhpx0"
ddgs.text()
```

The AI searches educational/tutorial websites such as:

* GeeksForGeeks
* Programming tutorials
* Technical articles

---

# 5. Code Extraction

After finding webpages, NSAI-K1 extracts:

* Paragraph explanations
* Code blocks
* Tutorials

Using:

```python id="55r08q"
BeautifulSoup
```

The system detects programming content using signatures like:

```python id="8im4z4"
def
print(
#include
public class
```

This allows the AI to fetch:

* Python code
* C++
* Java
* General programming examples

---

# 6. Wikipedia Fallback

If no useful coding/tutorial result is found, NSAI-K1 uses Wikipedia.

Using:

```python id="4pdkn0"
wikipedia.summary()
```

Purpose:

* Basic knowledge responses
* Lightweight factual information
* Educational fallback

---

# 7. Response Generation

The final response is displayed using:

```python id="0v5o8r"
slowprint()
```

or Streamlit chat rendering.

Example:

```text id="ejm1yj"
Bot: Python is a high-level programming language.
```

---

# 8. Memory Saving System

Useful responses are automatically stored into:

```bash id="4q9xyn"
data/dataset.jsonl
```

Only valid and non-duplicate responses are saved.

Purpose:

* Local memory growth
* Faster future responses
* Lightweight learning behavior

---

# Transformer Architecture

NSAI-K1 uses a custom Mini Transformer architecture.

Main components:

| Component            | Purpose                      |
| -------------------- | ---------------------------- |
| Token Embedding      | Converts tokens into vectors |
| Positional Embedding | Adds sequence understanding  |
| Self Attention       | Learns token relationships   |
| Feed Forward Layers  | Processes learned features   |
| LayerNorm            | Stabilizes training          |
| Softmax Sampling     | Generates output tokens      |

---

# Tokenizer System

NSAI-K1 uses a custom word-level tokenizer.

Features:

* Vocabulary generation
* Special tokens
* JSON vocabulary saving
* Encoding and decoding

Special Tokens:

| Token   | Purpose               |
| ------- | --------------------- |
| `<pad>` | Padding               |
| `<bos>` | Beginning of sequence |
| `<eos>` | End of sequence       |
| `<unk>` | Unknown token         |

---

# Training Workflow

Training process:

```text id="z0p56r"
Dataset
   ↓
Tokenizer Training
   ↓
Text Encoding
   ↓
Dataset Chunking
   ↓
Transformer Training
   ↓
Checkpoint Saving
```

---

# Checkpoints

After training, NSAI-K1 saves:

```bash id="9b6r9g"
mini_transformer.pt
tokenizer.json
config.json
```

These files are stored inside:

```bash id="4p9e4g"
checkpoints/
```

---

# Streamlit Interface

NSAI-K1 includes a lightweight Streamlit UI.

Features:

* Chat interface
* Session history
* Black and white theme
* Lightweight local deployment

Run using:

```bash id="8klfuh"
streamlit run app.py
```

---

# Current Limitations

NSAI-K1 is still a lightweight research project.

Current limitations:

* Small dataset (~3K lines)
* Limited reasoning
* Limited long-context memory
* Basic response generation
* Lightweight architecture only

---

# Future Plans (NSAI-K2)

Planned upgrades:

* ~30K dataset
* Better reasoning
* Improved memory system
* Faster inference
* Better Transformer scaling
* Advanced tokenizer
* Improved web research
* BEP Tech integration

---

# Author

Navtej Singh Saggar

NSAI-K1 is an experimental lightweight Transformer AI system focused on local research and development.
