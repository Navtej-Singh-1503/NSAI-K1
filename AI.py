'''
AUTHOR : NAVTEJ SINGH SAGGAR
VERSION : 5.4.0 (Memory Optimization & Error-Free)
NAME : NSAI-K1
'''

from __future__ import annotations
import argparse
import json
import re
import os
import time
import requests
import warnings
import wikipedia
import torch
import urllib.parse
from pathlib import Path
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from ddgs import DDGS

# Style Imports
try:
    from model import MiniTransformerLM, TransformerConfig
    from tokenizer import SimpleTokenizer
    from logo import logo
    from Style.slowprint import slowprint
except ImportError:
    def slowprint(text): print(text)
    logo = "--- NSAI-K1 ---"

warnings.filterwarnings("ignore", module='wikipedia')

######################################## Core Logic ########################################

def get_code_truth(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return None
        soup = BeautifulSoup(response.text, 'html.parser')

        # 1. Fetch Description
        description = ""
        for p in soup.find_all('p')[:3]:  # First 3 paragraphs
            text = p.get_text().strip()
            if len(text) > 50: description += text + " "

        # 2. Fetch Code
        code_blocks = soup.find_all(['pre', 'code'])
        best_code = ""
        sigs = ["def ", "print(", "input(", "#include", "iostream", "public class", "int main"]

        for block in code_blocks:
            content = block.get_text().strip()
            if any(s in content for s in sigs):
                content = re.sub(r'^\s*\d+[\.\s]*', '', content, flags=re.M)
                if len(content) > len(best_code): best_code = content

        if best_code:
            return f"{description[:500]}...\n\nCODE:\n{best_code}"
        return None
    except: return None

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def solve_math(user_input):
    match = re.search(r"(\d+(?:\s*[\+\-\*/]\s*\d+)+)", user_input)
    if match:
        try:
            res = eval(match.group(1), {"__builtins__": None}, {})
            return f"Calculation Result: {res}"
        except: return None
    return None

######################################## Main Loop ########################################

def chat_loop(args):
    # Initialize Model
    cfg = json.loads((Path(args.checkpoint_dir) / "config.json").read_text())
    tokenizer = SimpleTokenizer.load(Path(args.checkpoint_dir) / "tokenizer.json")
    model = MiniTransformerLM(TransformerConfig(**cfg)).to(args.device)
    model.load_state_dict(torch.load(Path(args.checkpoint_dir) / "mini_transformer.pt", map_location=args.device))
    model.eval()

    dataset_path = Path("data/dataset.jsonl")
    if not os.path.exists('data'): os.makedirs('data')

    # Load unique data into RAM
    all_data = []
    seen_prompts = set()
    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry["prompt"] not in seen_prompts:
                        all_data.append(entry)
                        seen_prompts.add(entry["prompt"])
                except: continue

    os.system('clear' if os.name == 'posix' else 'cls')
    print(logo)
    print("NSAI-K1 Online.\n")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input or user_input.lower() in ["exit", "quit"]: break

        assistant_reply = ""
        is_valuable = False # Should we save this to memory?

        # 1. RAM Search (Already in Dataset)
        for data in all_data:
            if similarity(data["prompt"], user_input) > 0.92:
                assistant_reply = data["response"]
                is_valuable = False # Don't save a duplicate
                break

        # 2. Math Engine
        if not assistant_reply:
            math_res = solve_math(user_input)
            if math_res:
                assistant_reply = math_res
                is_valuable = True

        # 3. Web & Code Search
        if not assistant_reply:
            code_keywords = ["code", "program", "script", "python", "cpp", "c++", "java"]
            is_code_req = any(kw in user_input.lower() for kw in code_keywords)

            try:
                with DDGS() as ddgs:
                    # Clean query for search
                    clean_q = user_input.replace("++", "cpp")
                    results = list(ddgs.text(f"{clean_q} tutorial site:geeksforgeeks.org", max_results=3))

                    for r in results:
                        web_data = get_code_truth(r['href'])
                        if web_data:
                            assistant_reply = web_data
                            is_valuable = True
                            break

                # Wikipedia Fallback
                if not assistant_reply and not is_code_req:
                    wiki = wikipedia.search(user_input)
                    if wiki:
                        assistant_reply = wikipedia.summary(wiki[0], sentences=3)
                        is_valuable = True

            except Exception as e:
                assistant_reply = f"System Error: Network connection issue."
                is_valuable = False

        # 4. Final Output and Memory Save
        if not assistant_reply:
            assistant_reply = "I couldn't find a clear answer. Try rephrasing for Class 9 logic."
            is_valuable = False

        slowprint(f"Bot: {assistant_reply}")

        # Save ONLY if it's a new, valid answer
        if is_valuable and user_input not in seen_prompts:
            with open(dataset_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"prompt": user_input, "response": assistant_reply}) + "\n")
            all_data.append({"prompt": user_input, "response": assistant_reply})
            seen_prompts.add(user_input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cpu")
    chat_loop(parser.parse_args())
