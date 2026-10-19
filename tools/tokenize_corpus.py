#!/usr/bin/env python3
"""Tokenise le corpus avec le tokenizer SentencePiece LLaMA (32 000 tokens).

Le même vocabulaire sera écrit dans le GGUF, ce qui garantit que llama.cpp
tokenise exactement comme l'entraînement.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
SPM = ROOT / "data" / "tokenizer_nanofr.model"
BOS, EOS = 1, 2


def encode_file(sp: spm.SentencePieceProcessor, path: Path) -> np.ndarray:
    ids: list[int] = []
    for doc in path.read_text(encoding="utf-8").split("\n\n"):
        doc = doc.strip()
        if not doc:
            continue
        ids.append(BOS)
        ids.extend(sp.encode(doc, out_type=int))
        ids.append(EOS)
    return np.asarray(ids, dtype=np.uint16)


def main() -> None:
    sp = spm.SentencePieceProcessor(model_file=str(SPM))
    for split in ("train", "val"):
        arr = encode_file(sp, CORPUS / f"{split}.txt")
        out = CORPUS / f"{split}.npy"
        np.save(out, arr)
        print(f"{split}: {len(arr)/1e6:.2f} M tokens -> {out} "
              f"({out.stat().st_size/1e6:.1f} Mo)")


if __name__ == "__main__":
    main()
