#!/usr/bin/env python3
"""Exporte un checkpoint NanoFR (numpy) en GGUF « llama ».

Les noms et l'orientation des tenseurs suivent la convention GGUF/PyTorch
(out, in) ; llama.cpp lit le fichier sans conversion.

Usage : python3 tools/export_gguf.py data/nanofr_ckpt.npz data/nanofr_f32.gguf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

sys.path.insert(0, "/home/user/llama.cpp/gguf-py")
from gguf import GGUFWriter, TokenType  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOK = ROOT / "data" / "tokenizer_nanofr.model"


def main() -> None:
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "nanofr_ckpt.npz"
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "nanofr_f32.gguf"

    ck = np.load(ckpt_path)
    sp = spm.SentencePieceProcessor(model_file=str(TOK))
    vocab = sp.get_piece_size()

    # déduire la config des tenseurs
    n_layers = 1 + max(int(k.split(".")[0][1:]) for k in ck.files if k.startswith("l"))
    d = ck["tok_embd"].shape[1]
    ffn = ck["l0.w_gate"].shape[1]
    n_heads_times_hd = ck["l0.wq"].shape[1]
    n_kv_times_hd = ck["l0.wk"].shape[1]
    n_heads, n_kv = 4, 2
    head_dim = n_heads_times_hd // n_heads
    ctx = 256

    w = GGUFWriter(str(out_path), "llama")
    w.add_name("NanoFR-6M")
    w.add_block_count(n_layers)
    w.add_context_length(ctx)
    w.add_embedding_length(d)
    w.add_feed_forward_length(ffn)
    w.add_head_count(n_heads)
    w.add_head_count_kv(n_kv)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_dimension_count(head_dim)
    w.add_rope_freq_base(10000.0)
    w.add_file_type(0)  # F32

    tokens = [sp.id_to_piece(i) for i in range(vocab)]
    scores = [float(sp.get_score(i)) for i in range(vocab)]
    types = []
    for i in range(vocab):
        if sp.is_unknown(i):
            types.append(TokenType.UNKNOWN)
        elif sp.is_control(i):
            types.append(TokenType.CONTROL)
        elif sp.is_byte(i):
            types.append(TokenType.BYTE)
        else:
            types.append(TokenType.NORMAL)
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(types)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)
    w.add_add_bos_token(True)

    w.add_tensor("token_embd.weight", np.ascontiguousarray(ck["tok_embd"]))
    w.add_tensor("output_norm.weight", np.ascontiguousarray(ck["out_norm"]))
    for i in range(n_layers):
        p = f"l{i}."
        g = f"blk.{i}."
        w.add_tensor(g + "attn_norm.weight", np.ascontiguousarray(ck[p + "attn_norm"]))
        w.add_tensor(g + "attn_q.weight", np.ascontiguousarray(ck[p + "wq"].T))
        w.add_tensor(g + "attn_k.weight", np.ascontiguousarray(ck[p + "wk"].T))
        w.add_tensor(g + "attn_v.weight", np.ascontiguousarray(ck[p + "wv"].T))
        w.add_tensor(g + "attn_output.weight", np.ascontiguousarray(ck[p + "wo"].T))
        w.add_tensor(g + "ffn_norm.weight", np.ascontiguousarray(ck[p + "ffn_norm"]))
        w.add_tensor(g + "ffn_gate.weight", np.ascontiguousarray(ck[p + "w_gate"].T))
        w.add_tensor(g + "ffn_up.weight", np.ascontiguousarray(ck[p + "w_up"].T))
        w.add_tensor(g + "ffn_down.weight", np.ascontiguousarray(ck[p + "w_down"].T))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    n_params = sum(int(v.size) for k, v in ck.items() if k != "step")
    print(f"écrit {out_path}  ({out_path.stat().st_size/1e6:.1f} Mo)")
    print(f"paramètres {n_params/1e6:.2f} M, couches {n_layers}, d={d}, "
          f"ffn={ffn}, têtes {n_heads}/{n_kv}, vocab {vocab}")


if __name__ == "__main__":
    main()
