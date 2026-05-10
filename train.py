from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import *
'''(
    LanguageModelingDataset,
    collect_training_texts,
    collate_lm_batch,
    load_jsonl_conversations,
)'''
from model import MiniTransformerLM, TransformerConfig
from tokenizer import SimpleTokenizer, TokenizerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local mini transformer chatbot")
    parser.add_argument("--data", type=str, default="data/dataset.jsonl")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--context_size", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    records = load_jsonl_conversations(args.data)
    texts = collect_training_texts(records)

    tokenizer = SimpleTokenizer.train_from_texts(texts, config=TokenizerConfig(min_freq=args.min_freq))
    dataset = LanguageModelingDataset(texts=texts, tokenizer=tokenizer, context_size=args.context_size)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_lm_batch(b, tokenizer.pad_id),
    )

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        context_size=args.context_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    model = MiniTransformerLM(config).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for x, y in loader:
            x = x.to(args.device)
            y = y.to(args.device)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        print(f"Epoch {epoch:02d}/{args.epochs} - loss: {avg_loss:.4f}"+" [+] TRAINING AI. IT WILL COMPLETED SOON!!")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "mini_transformer.pt"
    tok_path = out_dir / "tokenizer.json"
    cfg_path = out_dir / "config.json"

    torch.save(model.state_dict(), model_path)
    tokenizer.save(tok_path)
    cfg_path.write_text(json.dumps(config.__dict__, indent=2), encoding="utf-8")

    print(f"Saved model to: {model_path}")
    print(f"Saved tokenizer to: {tok_path}")
    print(f"Saved config to: {cfg_path}")


if __name__ == "__main__":
    main()
