#!/usr/bin/env python3
"""Entraîne un tokenizer SentencePiece français (BPE, 8192 pièces, byte fallback).

Pourquoi notre propre tokenizer plutôt que celui d'un modèle existant :
  * 8 192 pièces au lieu de 32 000 ou 262 000 -> la projection de sortie coûte
    4 à 32 fois moins cher, or c'est elle qui domine le calcul d'un petit modèle ;
  * aucune dépendance à une licence de modèle tiers ;
  * byte_fallback=True garantit que tout texte français reste encodable.
"""
from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus" / "train.txt"
OUT = ROOT / "data" / "tokenizer_nanofr"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(CORPUS),
        model_prefix=str(OUT),
        vocab_size=8192,
        model_type="bpe",
        byte_fallback=True,
        character_coverage=0.9995,
        num_threads=2,
        input_sentence_size=1_000_000,
        shuffle_input_sentence=True,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=-1,
        train_extremely_large_corpus=False,
    )
    sp = spm.SentencePieceProcessor(model_file=str(OUT) + ".model")
    print("vocab :", sp.get_piece_size())
    sample = "Il était une fois, à Paris, un homme qui ne croyait pas aux modèles de langue."
    ids = sp.encode(sample)
    print(f"{len(sample)} caractères -> {len(ids)} tokens")
    print(ids)
    print(sp.decode(ids))


if __name__ == "__main__":
    main()
