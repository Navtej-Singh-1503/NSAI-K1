from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


TOKEN_PATTERN = re.compile(r"[A-Za-z']+|[0-9]+|[^\w\s]")


@dataclass
class TokenizerConfig:
    min_freq: int = 2
    lowercase: bool = True


class SimpleTokenizer:
    """
    A beginner-friendly word-level tokenizer.

    It keeps punctuation as separate tokens and supports saving/loading
    vocabulary to JSON.
    """

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, stoi: dict[str, int], config: TokenizerConfig | None = None):
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}
        self.config = config or TokenizerConfig()

        self.pad_id = self.stoi[self.PAD]
        self.bos_id = self.stoi[self.BOS]
        self.eos_id = self.stoi[self.EOS]
        self.unk_id = self.stoi[self.UNK]

    @classmethod
    def train_from_texts(
        cls,
        texts: Iterable[str],
        config: TokenizerConfig | None = None,
    ) -> "SimpleTokenizer":
        config = config or TokenizerConfig()
        freq: dict[str, int] = {}

        for text in texts:
            for tok in cls._tokenize_text(text, lowercase=config.lowercase):
                freq[tok] = freq.get(tok, 0) + 1

        stoi = {
            cls.PAD: 0,
            cls.BOS: 1,
            cls.EOS: 2,
            cls.UNK: 3,
        }

        for token, count in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
            if count >= config.min_freq and token not in stoi:
                stoi[token] = len(stoi)

        return cls(stoi=stoi, config=config)

    @staticmethod
    def _tokenize_text(text: str, lowercase: bool = True) -> List[str]:
        normalized = text.strip()
        if lowercase:
            normalized = normalized.lower()
        return TOKEN_PATTERN.findall(normalized)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self._tokenize_text(text, lowercase=self.config.lowercase)
        ids = [self.stoi.get(tok, self.unk_id) for tok in tokens]
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        specials = {self.PAD, self.BOS, self.EOS}
        tokens: list[str] = []

        for i in ids:
            tok = self.itos.get(int(i), self.UNK)
            if skip_special_tokens and tok in specials:
                continue
            tokens.append(tok)

        # basic readable detokenization
        text = " ".join(tokens)
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        return text.strip()

    def save(self, path: str | Path) -> None:
        payload = {
            "stoi": self.stoi,
            "config": {
                "min_freq": self.config.min_freq,
                "lowercase": self.config.lowercase,
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SimpleTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = TokenizerConfig(**payload.get("config", {}))
        return cls(stoi=payload["stoi"], config=cfg)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)
