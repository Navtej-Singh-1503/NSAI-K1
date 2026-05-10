from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
from torch.utils.data import Dataset

from tokenizer import SimpleTokenizer


# ✅ CLEAN TAGS (no "User/Assistant" confusion)
PROMPT_TAG = "Q:"
RESPONSE_TAG = "A:"


@dataclass
class ConversationRecord:
    prompt: str
    response: str


# ✅ LOAD DATA (strict + clean)
def load_jsonl_conversations(path: str | Path) -> List[ConversationRecord]:
    records: list[ConversationRecord] = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)

                p = obj.get("prompt")
                r = obj.get("response")

                if not p or not r:
                    continue

                # ✅ clean strings
                p = str(p).strip()
                r = str(r).strip()

                if len(p) < 2 or len(r) < 2:
                    continue

                records.append(ConversationRecord(prompt=p, response=r))

            except json.JSONDecodeError:
                continue

    return records


# ✅ CLEAN FORMAT (VERY IMPORTANT)
def format_conversation(record: ConversationRecord) -> str:
    return f"{PROMPT_TAG} {record.prompt}\n{RESPONSE_TAG} {record.response}\n"


# ✅ COLLECT TEXTS
def collect_training_texts(records: Iterable[ConversationRecord]) -> list[str]:
    texts = [format_conversation(r) for r in records]

    # ✅ shuffle for better learning
    import random
    random.shuffle(texts)

    return texts


# ✅ DATASET CLASS
class LanguageModelingDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, context_size: int):
        self.examples: list[torch.Tensor] = []

        for text in texts:
            token_ids = tokenizer.encode(text, add_special_tokens=True)

            # ❌ skip too short
            if len(token_ids) < 5:
                continue

            # ✅ better chunking (avoid breaking too aggressively)
            for i in range(0, len(token_ids) - 1, context_size // 2):
                chunk = token_ids[i : i + context_size + 1]

                if len(chunk) < 5:
                    continue

                self.examples.append(torch.tensor(chunk, dtype=torch.long))

        if not self.examples:
            raise ValueError("No training data created. Check dataset quality.")


    def __len__(self) -> int:
        return len(self.examples)


    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.examples[idx]


# ✅ COLLATE FUNCTION (same but clean)
def collate_lm_batch(batch: list[torch.Tensor], pad_id: int):
    max_len = max(x.size(0) for x in batch)

    x = torch.full((len(batch), max_len - 1), pad_id, dtype=torch.long)
    y = torch.full((len(batch), max_len - 1), -100, dtype=torch.long)

    for i, sample in enumerate(batch):
        inp = sample[:-1]
        tgt = sample[1:]

        x[i, : inp.size(0)] = inp
        y[i, : tgt.size(0)] = tgt

    return x, y