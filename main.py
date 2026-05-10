'''
Author : Navtej-Singh-Saggar
Version : 5.7.9

Please Check

- README.md
- LICENSE
- Guide.md (optional)

'''



import streamlit as st
import json
import torch
import os
from pathlib import Path
from AI import solve_math, get_code_truth, similarity, MiniTransformerLM, TransformerConfig, SimpleTokenizer
from ddgs import DDGS
import wikipedia

# --- Configuration ---
CHECKPOINT_DIR = "checkpoints"
DEVICE = "cpu"
DATASET_PATH = Path("data/dataset.jsonl")

# --- Page Setup ---
st.set_page_config(page_title="NSAI-K1", page_icon="🤖", layout="centered")

# Custom CSS for the "Black & White" UI look from your image
st.markdown("""
    <style>
    .main { background-color: #000000; color: #ffffff; }
    .stTextInput > div > div > input {
        background-color: #1e1e1e;
        color: white;
        border: 2px solid #ffffff;
        border-radius: 25px;
    }
    .stChatMessage { border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- Logic Initialization ---
@st.cache_resource
def load_model_and_data():
    # Load Model
    cfg = json.loads((Path(CHECKPOINT_DIR) / "config.json").read_text())
    tokenizer = SimpleTokenizer.load(Path(CHECKPOINT_DIR) / "tokenizer.json")
    model = MiniTransformerLM(TransformerConfig(**cfg)).to(DEVICE)
    model.load_state_dict(torch.load(Path(CHECKPOINT_DIR) / "mini_transformer.pt", map_location=DEVICE))
    model.eval()

    # Load Memory
    all_data = []
    seen_prompts = set()
    if DATASET_PATH.exists():
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry["prompt"] not in seen_prompts:
                        all_data.append(entry)
                        seen_prompts.add(entry["prompt"])
                except: continue
    return model, tokenizer, all_data, seen_prompts

model, tokenizer, all_data, seen_prompts = load_model_and_data()

# --- UI Header ---
st.title("NSAI-K1")
st.caption("Author : Navtej-Singh-Saggar")
st.write("---")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input & Logic ---
if prompt := st.chat_input("What's on Today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = ""
        is_valuable = False

        # 1. Memory Search
        for data in all_data:
            if similarity(data["prompt"], prompt) > 0.92:
                response = data["response"]
                break

        # 2. Math
        if not response:
            math_res = solve_math(prompt)
            if math_res:
                response = math_res
                is_valuable = True

        # 3. Web Search
        if not response:
            try:
                with DDGS() as ddgs:
                    clean_q = prompt.replace("++", "cpp")
                    results = list(ddgs.text(f"{clean_q} tutorial site:geeksforgeeks.org", max_results=2))
                    for r in results:
                        web_data = get_code_truth(r['href'])
                        if web_data:
                            response = web_data
                            is_valuable = True
                            break

                if not response:
                    wiki = wikipedia.search(prompt)
                    if wiki:
                        response = wikipedia.summary(wiki[0], sentences=3)
                        is_valuable = True
            except:
                response = "Connection error. Please check your network."

        if not response:
            response = "I couldn't find a clear answer. Try rephrasing for Class 9 logic."

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Save to memory if valuable
        if is_valuable and prompt not in seen_prompts:
            if not os.path.exists('data'): os.makedirs('data')
            with open(DATASET_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
            all_data.append({"prompt": prompt, "response": response})
            seen_prompts.add(prompt)