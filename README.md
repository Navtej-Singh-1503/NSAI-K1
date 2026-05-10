# NSAI-K1

Lightweight Transformer-based AI Assistant built using PyTorch.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-5.7.9-purple)

---

# About

NSAI-K1 is a custom Transformer-based AI assistant created by Navtej Singh Saggar.

Currently, the project mainly works as a:
- Web research assistant
- Lightweight coding helper
- Experimental local AI system

Due to the current small dataset size (~3K lines), the model is still under active development and research.

Future versions will include:
- Larger datasets
- Better reasoning
- Improved memory handling
- More accurate responses
- Better conversational quality

The AI can:

- Solve mathematics
- Search educational and technical content
- Fetch programming tutorials
- Learn from previous conversations
- Use Wikipedia for knowledge fallback
- Run fully local using PyTorch

---

# Features

## Transformer-Based AI

Custom-built Mini Transformer Language Model using PyTorch.

## Smart Memory System

Stores useful conversations into dataset memory automatically.

## Hybrid Search Intelligence

Uses:
- DuckDuckGo Search (`ddgs`)
- Wikipedia
- Web scraping via BeautifulSoup

## Lightweight

Runs on CPU with low memory usage.

## Streamlit UI

Modern black and white chatbot interface.

## Local AI Research Project

NSAI-K1 is designed as an experimental lightweight AI architecture for learning and research purposes.

---

# Project Structure

```bash
NSAI-K1/
│
├── AI.py
├── train.py
├── model.py
├── tokenizer.py
├── dataset.py
├── app.py
│
├── checkpoints/
│   ├── mini_transformer.pt
│   ├── tokenizer.json
│   └── config.json
│
├── data/
│   └── dataset.jsonl
│
├── Style/
│   └── slowprint.py
│
├── requirements.txt
├── README.md
├── LICENSE
└── Guide.md
├── Help/
|     └── Train.md
|     └── Working.md
├── img/
     └── img1.png
     └── img2.png
     └── img3.png
     └── img4.png
     └── img5.png
     └── img6.png
     └── img7.png
```
## Installation
### Step 1 : Clone Repository
```
git clone https://github.com/Navtej-Singh-1503/NSAI-K1.git
cd NSAI-K1
```

### Step 2 : Install libraries
```
python3 -m pip install -r requirements.txt
```

### Step 3 : Train AI
- Check Help/Train.md

### Step 4 : Use AI

For Terminal version
```
python3 AI.py
```
For Web version
```
streamlit run main.py
```

---
# Add-ons
## NSAI-K2
Upcoming improvements planned for NSAI-K2:

- ~30K lines dataset
- Better memory optimization
- Faster response generation
- Improved Transformer architecture
- Better contextual understanding
- Advanced web research system
- Cleaner code generation
- BEP Tech integration
- Many More

# Author
## Navtej Singh Saggar
- https://github.com/Navtej-Singh-1503

Focused on:
- AI Development
- Cyber Security
- Education Technology
- Lightweight AI Systems

# License

This project is licensed under the MIT License.

See LICENSE for more information.

# Support

If you like this project:

- Star the repository
- Fork the project
- Share with developers
- Contribute improvements

## Support My Work

You can support me via cryptocurrency:

- LTC : ltc1qspfztcvax7g9caqgdmp3ex6fytrr0dlssr0r45
- URL : litecoin:LTC1QSPFZTCVAX7G9CAQGDMP3EX6FYTRR0DLSSR0R45

# Owner

### Creater - Navtej-Singh-Saggar

- navtejsingh1503@gmail.com
- DOB - 15 March 2011
