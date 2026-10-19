# Références

**Compilé par :** KAIROS · **Date :** 2026-10-19
Chaque entrée indique **ce qu'elle soutient** dans nos documents. Une référence
qui ne sert à aucun chiffre est du décor ; elle n'a pas sa place ici.

---

## 1. Quantification ternaire et noyaux CPU

| Réf | Ce qu'elle établit | Lien |
|---|---|---|
| **BitNet b1.58 / bitnet.cpp** | noyaux ternaires *lossless* ; 2,37-6,17× (x86) et 1,37-5,07× (ARM) ; −71,9 à −82,2 % d'énergie (x86) ; démo 100 B sur un seul CPU à 5-7 tok/s | <https://arxiv.org/html/2410.16144v1> · <https://github.com/microsoft/BitNet> |
| **BitNet 2B4T, chiffres publics** | 0,4 Go hors embeddings, 29 ms/token, 0,028 J/token, MMLU 53,17 ; les gains n'apparaissent **que** via bitnet.cpp, pas via transformers | <https://tinyweights.dev/posts/1-bit-llms-bitnet-ternary-weights/> |
| **Statut de la démo 100 B** | démonstration de faisabilité sur un modèle **synthétisé**, pas un artefact public benchmarké | <https://www.continuumresourcesllc.com/continuum-edge-ai-bitnet-whitepaper.html> |
| **KTransformers (SOSP'25)** | 671 B Q4_K_M sur 1 GPU 24 Go + 382 Go de DRAM ; ~13,6-14 tok/s en décodage, jusqu'à 286 tok/s en prefill ; noyaux AMX/AVX-512 INT4 | <https://github.com/kvcache-ai/ktransformers> · [papier](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) |
| **KTransformers, contre-mesures** | expert-offload = 1,62× le décodage CPU pur, mais 22× plus lent que full-GPU ; AMX vs AVX2 = 1,38× ; AMX INT4 vs BF16 = 3,9× | <https://github.com/hyojunguy/ktransformers-moe-offload-bench> |
| **Débits réels 671 B** | M3 Ultra 192 Go ~17-18 tok/s ; RTX 4090 + 382 Go ~14 tok/s ; bi-EPYC ~5-8 tok/s | <https://groundy.com/articles/running-deepseek-r1-locally-hardware-requirements-quantization-and-real-throughput/> |

## 2. Architectures de référence

| Réf | Ce qu'elle établit | Lien |
|---|---|---|
| **Qwen3-Next-80B-A3B** | 80 B total / 3 B actifs (3,9 B selon NVIDIA) ; 48 couches = 12 × (3 Gated DeltaNet + 1 Gated Attention) ; 512 experts top-10 + 1 partagé ; contexte 262 144 natif ; 15 T tokens ; ~10× moins cher à entraîner | <https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct> · [NVIDIA](https://docs.api.nvidia.com/nim/reference/qwen-qwen3-next-80b-a3b-instruct) · [blog](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd) |
| **Qwen3-Next, analyse** | densité d'activation 3,7 % ; la loss baisse quand on augmente les paramètres experts à top-k fixe | <https://news.smol.ai/frozen-issues/25-09-11-qwen3-next> |
| **gpt-oss-120b** | 117 B / 5,1 B actifs, 36 couches, 128 experts top-4, MXFP4 à 4,25 bits, ~60,8 GiB, MMLU 90,0 | <https://openai.com/index/introducing-gpt-oss/> · <https://huggingface.co/openai/gpt-oss-120b> · [analyse](https://inferencex.semianalysis.com/model/gptoss-120b) |
| **Galerie d'architectures** | comparaison visuelle des blocs (utile pour les schémas) | <https://sebastianraschka.com/llm-architecture-gallery/> |

## 3. Physique de l'inférence

| Réf | Ce qu'elle établit | Lien |
|---|---|---|
| **Roofline, formalisation** | Williams, Waterman & Patterson (2009) ; point de crête = FLOPS / bande passante | <https://dl.acm.org/doi/10.1145/1498765.1498785> |
| **Roofline appliquée au décodage** | intensité arithmétique ≈ batch size ; ~1 FLOP/octet à batch 1 ; crête A100 ≈ 156, H100 ≈ 280-300 | <https://www.spheron.network/blog/the-roofline-model-for-llm-inference-are-you-compute-bound-o/> · <https://inferenceengineering.tech/learn/gpu-inference/> |
| **Débit = BW / octets par token** | exemple chiffré : 70 B fp16 = 140 Go → ~48 tok/s max sur 2×H100 | <https://www.sahilcreates.dev/blog/inference-deep-dive> |
| **Outil de diagnostic** | implémentation pratique de `tok/s = BW_eff / bytes_per_token` | <https://github.com/Pluenet-Killian/llm-roofline> |

## 4. KV cache

| Réf | Ce qu'elle établit | Lien |
|---|---|---|
| **TurboQuant** | 3-3,5 bits/coordonnée, ≥ 6× vs FP16, jusqu'à 8× de speedup attention, sans réentraînement, ICLR 2026 (Google Research + NYU) | <https://arxiv.org/abs/2504.19874> · [InfoQ](https://www.infoq.com/news/2026/04/turboquant-compression-kv-cache/) |
| **Survey KV 2026** | taxonomie éviction / compression / hybride ; KVQuant jusqu'à 10 M de contexte | <https://arxiv.org/html/2603.20397v1> |
| **kvtc** | jusqu'à 20× de compression (40× dans certains cas), ICLR 2026 | <https://arxiv.org/pdf/2511.01815> |
| **KIVI / KVQuant / CommVQ** | KIVI : 2 bits, 2,6× de mémoire au pic ; CommVQ : 1-2 bits, −87,5 % | <https://www.marktechpost.com/2026/04/29/top-10-kv-cache-compression-techniques-for-llm-inference-reducing-memory-overhead-across-eviction-quantization-and-low-rank-methods/> · <https://pith.science/paper/2506.18879> |
| **NVFP4 KV** | −50 % vs FP8, < 1 % de perte | <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/> |

## 5. Décodage spéculatif

| Réf | Ce qu'elle établit | Lien |
|---|---|---|
| **EAGLE-3** | 3,0-6,5× (6,65× à T=0), +20-40 % vs EAGLE-2 ; draft head d'une couche, ~5 % des paramètres | <https://arxiv.org/html/2503.01840v1> |
| **Comparaison EAGLE-3 / Medusa / Lookahead** | Medusa 2,2-3,6× ; Lookahead 1,5-2× | <https://iotdigitaltwinplm.com/speculative-decoding-eagle-3-vs-medusa-vs-lookahead-llm-inference-speedup-2026/> |
| **Acceptation en production** | 0,82-0,88 à batch 1, mais 0,52-0,62 à batch 64 (~breakeven 0,9-1,2×) ; `num_speculative_tokens` 5 par défaut, 8 pour le code | <https://www.spheron.network/blog/eagle-3-speculative-decoding-gpu-cloud/> |

## 6. Capacité sans calcul : couches mémoire

| Réf | Ce qu'elle établit | Lien |
|---|---|---|
| **Memory Layers at Scale (Meta)** | jusqu'à **128 B** de paramètres mémoire ; un 1,3 B avec mémoire ≈ Llama2-7B avec 10× moins de FLOPs ; +100 % en QA factuelle ; clés produit en O(√N) | <https://arxiv.org/abs/2412.09764> |
| **Product Keys (Lample et al.)** | lookup en O(√|K|) au lieu de O(|K|) | <https://ar5iv.labs.arxiv.org/html/1907.05242> |
| **UltraMemV2** | 120 B total / 2,5 B activés ; la densité d'activation compte plus que le nombre de paramètres sparse | <https://arxiv.org/pdf/2508.18756> |
| **Ultra-Sparse Memory Network** | la loss décroît linéairement quand les paramètres sparse croissent exponentiellement ; temps d'inférence constant | <https://arxiv.org/pdf/2411.12364> |
| **Fast-weight Product Key Memory** | mémoire épisodique mise à jour à l'inférence (TTT sur slots) | <https://arxiv.org/pdf/2601.00671> |

## 7. Entraînement à petit budget

| Réf | Ce qu'elle établit | Lien |
|---|---|---|
| **Chinchilla** | `L = E + A/N^α + B/D^β` ; entraînement compute-optimal ≈ 20 tokens/paramètre | <https://arxiv.org/abs/2203.15556> |
| **Sparse Upcycling** | ~50 % du budget initial suffit à battre le dense ; < 40 % de budget additionnel | <https://arxiv.org/pdf/2212.05055> |
| **Llama3-8B → MoE** | 8 experts top-2, **< 1 %** du compute de pré-entraînement (11 k GPU-h), +2 pts MMLU, 46,8 % MFU | <https://arxiv.org/html/2412.09952v1> |
| **Expert Upcycling** | 7 B → 13 B, 32 % d'heures GPU économisées, top-k inchangé | <https://arxiv.org/html/2604.19835v1> |
| **Recyclage de checkpoints 70 B** | +10,66 % d'accuracy vs from-scratch à budget égal | <https://arxiv.org/html/2510.08008v1> |

## 8. Entraînement décentralisé

| Réf | Ce qu'elle établit | Lien |
|---|---|---|
| **DiLoCo** | 8 travailleurs = qualité synchrone avec **500×** moins de communication | <https://arxiv.org/abs/2311.08105> |
| **Streaming DiLoCo** | 4 bits, 400× moins d'octets, 8× moins de bande passante au pic, loss 2,45 = 2,45 (1 B / 250 B tokens) | <https://arxiv.org/html/2501.18512v1> |
| **DiLoCoX** | pré-entraînement d'un **107 B** sur un lien **1 Gbit/s** ; 357× vs AllReduce ; 90-95 % d'utilisation | <https://arxiv.org/html/2506.21263v1> |
| **OpenDiLoCo** | implémentation ouverte ; int8 all-reduce, 400× (jusqu'à 2 000×) | <https://arxiv.org/pdf/2407.07852> |
| **INTELLECT-1** | 10 B paramètres entraînés sur 3 continents | <https://arxiv.org/html/2412.01152v1> |

## 9. Matériel (relevés d'octobre 2026)

| Réf | Ce qu'elle établit | Lien |
|---|---|---|
| **Strix Halo, prix** | 128 Go : 3 449-3 879 $ selon le constructeur ; 256 Go/s théoriques | <https://computingforgeeks.com/ryzen-ai-max-395-mini-pc-comparison/> |
| **Strix Halo, bande passante réelle** | ~215 Go/s mesurés (vs 256 théoriques) | <https://datahardware.ai/blog/amd-ai-max-395-explained> |
| **Mac Studio, specs officielles** | M4 Max 546 Go/s ; M3 Ultra 819 Go/s | <https://support.apple.com/en-us/122211> |
| **Mac Studio, prix 2026** | M4 Max 128 Go ~3 500 $ ; M3 Ultra 96 Go 5 299 $ ; M5 Ultra 1,2 To/s à partir de 5 499 $ | <https://insiderllm.com/guides/mac-studio-local-ai-workstation/> · <https://www.macrumors.com/guide/2025-vs-2026-mac-studio/> |

## 10. Sources internes

| Réf | Ce qu'elle établit |
|---|---|
| **Document partagé** `13-LF4vy1dyeh3JyWM8hHw5aoKJz1fpf5OeaUXn7CzjA` | cadrage 100 Mo, onglets 00-11, protocole W1-W6, ADR-001→005, chiffres NanoFR-75M |
| **Branche `arena/9c2ae177-scienceai`** | rapport `docs/RECHERCHE-LLM-100MB.md` et `tools/llm_budget.py` — **pas encore poussés** au moment de la rédaction |
| **`results/bwprobe-sandbox.md`** | mesure de localité mémoire (ce dépôt) |
| **`results/*.md`** | tous les budgets calculés (ce dépôt) |

---

## 11. Ce qui manque (à chercher)

1. **Une loi de puissance qualité vs bits** à taille constante. Personne ne l'a
   publiée. C'est l'expérience E-BIT, et c'est le verrou scientifique du projet.
2. **Des benchmarks de modèles ternaires > 3 B.** Il n'y en a pas publiquement.
   Toute affirmation « un 300 B ternaire vaut un 150 B 4 bits » est une
   extrapolation, pas un fait.
3. **Le coût réel du runtime** pour un modèle de 80 Go sur CPU : personne ne
   publie de RSS détaillé.
4. **Le débit des noyaux ternaires sur AVX-512 VNNI et AMX** pour des matrices
   de 1 408 × 3 584 : les chiffres publiés concernent surtout des modèles denses.
5. **La stabilité du routage** lors d'un upcycling ×4 sur un modèle déjà entraîné.
