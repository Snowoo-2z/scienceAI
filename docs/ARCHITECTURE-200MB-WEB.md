# Project LLM v0.3 — un généraliste dans 200 Mo, et comment le déployer

**Date** : 2026-10-19
**Statut** : révision après vérification des tailles GGUF et de la configuration réelle de Gemma 3 270M
**Remplace** : v0.2 (`docs/ARCHITECTURE-200MB-WEB.md` version précédente) et les recommandations de
`RECHERCHE-LLM-100MB.md` §0, §6, §9
**Répond à** : Silex (W3), MNÉMOSYNE (working set), KAIROS (métrologie)

---

## 0. Verdict

**Oui, un généraliste tient dans 200 Mo — à une condition précise : quantifier aussi les embeddings.**

Gemma 3 270M Instruct, checkpoint QAT INT4, avec les embeddings en 4 bits :

```
poids           150 Mo   [calculé ; 145–151 selon la taille des blocs]
KV (8 k, q8)     16,5 Mo
runtime          26 Mo
──────────────────────────
RSS             ~193 Mo   ✅ sous 200 Mo
```

C'est un vrai modèle généraliste : 6 000 Md de tokens d'entraînement, 140+ langues (dont le
français), 32 k de contexte natif, fonction d'appel d'outils.

### Les deux erreurs corrigées depuis la v0.1

| Affirmation initiale | Réalité mesurée |
|---|---|
| « Plancher d'un généraliste : 300–500 Mo » | **Faux.** 270 M à 4 bits = ~150 Mo. Le plancher était surestimé d'un facteur 2. |
| « Gemma 3 270M fait 253 Mo, hors budget » | **Vrai pour le GGUF standard, faux pour la QAT.** Le Q4_K_M garde les embeddings à ~8 bits : 170 M de paramètres = 170 Mo à eux seuls. En 4 bits partout, le fichier tombe à ~150 Mo. |

**La leçon : à cette échelle, le vocabulaire est le poste de coût dominant, et sa précision est un
choix de déploiement, pas une fatalité.** Gemma 3 a 262 144 tokens × 640 dimensions = 167,8 M
paramètres d'embeddings, soit 62 % des paramètres du modèle.

---

## 1. Le budget, en détail

| Configuration | Fichier | KV 2 k | KV 8 k | **RSS 2 k** | **RSS 8 k** | Verdict |
|---|---:|---:|---:|---:|---:|---|
| **Gemma 3 270M QAT, emb 4 bits** | **~150 Mo** | 7,1 | 16,5 | **183 Mo** | **193 Mo** | ✅ **recommandé** |
| Gemma 3 270M Q4_K_M (GGUF standard) | 253 Mo | 7,1 | 16,5 | 286 Mo | 296 Mo | ❌ (embeddings à 8 bits) |
| Gemma 3 270M Q2_K | 237 Mo | 7,1 | 16,5 | 270 Mo | 280 Mo | ❌ |
| Gemma 3 270M, vocabulaire élagué à 32 k | ~106 Mo | 4,1 | 16,5 | 139 Mo | 149 Mo | ✅ mais chirurgie + perte multilingue |
| MobileLLM-R1-140M Q4_K_M | 132 Mo | 11,8 | 47,2 | 170 Mo | 205 Mo | ✅ 2–4 k, mais spécialisé maths/code |
| SmolLM2-135M Q4_K_M | 105 Mo | 23,6 | 94,4 | 155 Mo | 225 Mo | ✅ 4 k max, anglais, faible |
| SmolLM2-360M Q4_K_M | 271 Mo | 41,9 | — | 339 Mo | — | ❌ |

Le KV de Gemma est remarquablement petit : **512 octets par token et par couche**, une seule tête
KV, head_dim 256, et surtout **15 couches sur 18 à fenêtre glissante de 512 tokens**. À 8 k de
contexte, il ne coûte que 16,5 Mo — trois fois moins que SmolLM2-135M, qui a pourtant deux fois
moins de paramètres.

Conséquence directe : **on peut offrir du contexte.** 8 k tient ; 16 k est à la limite (205 Mo) ;
32 k est hors budget sans élagage du vocabulaire.

---

## 2. La recette de quantification (le point technique central)

Le checkpoint officiel est `google/gemma-3-270m-it-qat-q4_0-unquantized` : des poids « non
quantifiés » mais **entraînés pour supporter l'INT4**. C'est ce qui rend la quantification des
embeddings sûre — sans QAT, passer les embeddings en 4 bits casse le modèle.

```bash
# 1. Convertir le checkpoint QAT en GGUF (F16 intermédiaire)
python convert_hf_to_gguf.py gemma-3-270m-it-qat-q4_0-unquantized --outfile gemma-qat-f16.gguf --outtype f16

# 2. Quantifier TOUT en Q4_0, embeddings compris
./llama-quantize \
    --token-embedding-type q4_0 \
    --output-tensor-type q4_0 \
    gemma-qat-f16.gguf gemma-3-270m-qat-q4_0-emb4.gguf Q4_0

# 3. Vérifier la taille (attendu : 145–151 Mo) et la perplexité
ls -l gemma-3-270m-qat-q4_0-emb4.gguf
```

**Pourquoi si peu de gens le font :** les quantifications « clés en main » (Q4_K_M, Q4_K_S)
protègent les embeddings parce que c'est le choix sûr pour la qualité. À 270 M avec un vocabulaire
de 262 k, ce choix coûte 100 Mo. La QAT existe précisément pour lever cette contrainte.

**Risque à mesurer (E1) :** la perte de qualité après quantification des embeddings. Critère :
perplexité sur un texte français de référence en hausse de **≤ 5 %**, et baisse d'exactitude
**≤ 3 points** sur le jeu de 500 requêtes (`docs/PROTOCOLE-EVALUATION.md`).

---

## 3. Entraînement : ce que 1 Go permet

### 3.1 La règle

> **1 Go permet d'adapter un modèle jusqu'à ~300 M, ou de pré-entraîner un modèle jusqu'à ~60 M.
> Il ne permet pas de pré-entraîner un 270 M.**

| Méthode sur Gemma 3 270M | Poids | Optimiseur | Total | Sous 1 Go ? |
|---|---:|---:|---:|---|
| Full fine-tuning, Adam fp32 | 1,07 Go | 3,2 Go | 4,3 Go | ❌ |
| Full fine-tuning, Adam 8-bit | 1,07 Go | 1,6 Go | 2,7 Go | ❌ |
| LoRA bf16 | 0,54 Go | ~0,06 Go | 0,60 Go | ⚠️ avec activations, non |
| **QLoRA NF4** | **0,134 Go** | **~0,03 Go** | **0,16 Go** | ✅ |

Avec les activations (batch 1, séquence 1 024, gradient checkpointing), on arrive à **350–500 Mo sur
CPU** et **600–900 Mo sur GPU 1 Go** (le contexte CUDA coûte 200–400 Mo à lui seul).

### 3.2 Ce que ça coûte en temps

| Run | FLOPs | CPU 30 GFLOPS | GPU T4 | Coût |
|---|---:|---:|---:|---:|
| QLoRA SFT, 5 M tokens | 8,0 × 10¹⁵ | ~3 jours | ~20 min | < 0,50 $ |
| QLoRA distillation, 20 M tokens | 3,2 × 10¹⁶ | ~12 jours | ~1,1 h | ~1 $ |
| Pré-entraînement 60 M, 1,2 Md tokens | 4,3 × 10¹⁷ | ~166 jours | ~15 h | ~10 $ |

**Conclusion :** un QLoRA sérieux est un run de quelques heures pour 1–2 $. Le pré-entraînement
from scratch, même à 60 M, coûte 166 jours de CPU — et donne un modèle très inférieur à Gemma 3 270M.

### 3.3 La chaîne d'adaptation

```
1. Base     : gemma-3-270m-it-qat-q4_0-unquantized (poids QAT, robustes à l'INT4)
2. Données  : 5–20 M tokens de français + domaine (instructions, format, refus)
3. QLoRA    : NF4, r=16, alpha=32, cibles q/k/v/o + gate/up/down
              seq 1 024, batch 1, gradient checkpointing, LR 5e-5 à 1e-4
4. Fusion   : merge des adaptateurs
5. Export   : Q4_0 avec --token-embedding-type q4_0  →  ~150 Mo
6. Mesure   : perplexité, jeu de 500 requêtes, RSS réel — avant mise en ligne
```

Partir du checkpoint QAT plutôt que du fp16 évite de cumuler la perte du fine-tuning et celle de la
quantification. Attention : un LR trop élevé dégrade la robustesse QAT. Vérifier la perplexité après
export ; si la hausse dépasse 5 %, refaire un court QAT de récupération.

---

## 4. Déploiement web sur petits PC

### 4.1 Architecture

```
Navigateur ── HTTPS + SSE ──► Caddy / nginx (TLS, rate limit)
                                    │
                                    ├──► cache applicatif (SQLite, réponses fréquentes)
                                    ▼
                            llama-server (API OpenAI)
                                    │  --ctx-size 8192 --parallel 4 --cont-batching
                                    ▼
                            GGUF en mmap, lecture seule
```

Un seul processus par nœud. Le KV cache est un **pool partagé** : `--ctx-size` est le total pour
tous les slots, pas la taille par slot. `--parallel 4 --ctx-size 8192` donne ~2 048 tokens par
requête.

### 4.2 Configuration de référence

```bash
llama-server \
  --model /models/gemma-3-270m-qat-q4_0-emb4.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 8192 --parallel 4 --cont-batching \
  --threads 4 --batch-size 1024 --ubatch-size 256 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --metrics --api-key file:/etc/llama/api-key
```

| Poste | Valeur |
|---|---|
| Poids | 150 Mo |
| KV (8 k total, q8) | 16,5 Mo |
| Runtime + scratch | 26 Mo |
| **RSS** | **~193 Mo** |
| Slots | 4 × 2 048 tokens |
| RSS par visiteur supplémentaire | **0** (le KV est déjà alloué) |

**Placer le service sous `MemoryMax=220M`** (systemd) : si le processus dépasse, c'est un bug de
configuration, pas une fatalité.

### 4.3 Matériel

| Plateforme | RAM | Débit mono-flux [estim.] | Visiteurs confortables | Puissance |
|---|---:|---:|---:|---:|
| Raspberry Pi 5 | 8 Go | 5–10 tok/s | 1–2 | 5–8 W |
| Mini-PC N100 | 16 Go | 12–20 tok/s | 3–4 | 10–15 W |
| Mini-PC Ryzen 5 5500U / i5-1235U | 16 Go | 25–40 tok/s | 6–10 | 15–25 W |
| Ancien laptop i5-8250U | 8–16 Go | 15–25 tok/s | 4–8 | 15 W |

**Pas de GPU.** À 270 M de paramètres, le décodage est limité par la bande passante mémoire, pas par
le calcul. Un GPU n'apporte presque rien et coûte 10× plus cher.

### 4.4 Capacité et montée en charge

Un nœud N100 à 4 slots et ~30 tok/s agrégés traite **~1 requête toutes les 7 s**, soit ~12 000
requêtes/jour en pointe continue. Au-delà, on ajoute des réplicas **sans état** derrière nginx : le
modèle est identique, la session est côté client, rien à synchroniser.

### 4.5 Énergie

1 000 requêtes/jour de 200 tokens sur un N100 à 15 W : ~5,6 h de calcul, **~84 Wh/jour**, soit
**~8 €/an** à 0,25 €/kWh. Le serveur coûte plus cher en veille qu'en calcul : réveil à la demande et
veille nocturne sont les vrais leviers.

### 4.6 Sécurité

Clé API sur llama-server (jamais exposé directement), rate limit par IP (nginx `limit_req`),
entrée ≤ 2 000 caractères, sortie ≤ 200–300 tokens, timeout 60 s, prompt système court
(200–400 tokens), historique tronqué aux 2–3 derniers tours, filtre de sortie, alerte si RSS > 200 Mo.

### 4.7 Le reste de la pile (à ne pas oublier)

| Poste | RSS [estimé] |
|---|---:|
| llama-server + modèle | 193 Mo |
| Caddy | 20 Mo |
| Application (Go recommandé) | 20 Mo |
| Cache SQLite + FTS5 (BM25) | 10 Mo |
| Noyau Linux + systemd | 300–600 Mo |

**« 200 Mo » désigne le processus d'inférence, pas la machine.** Un noyau Linux consomme à lui seul
300–600 Mo. Pour le RAG, **BM25/FTS5 plutôt qu'un index vectoriel** : zéro modèle, ~10 Mo, et de
meilleurs résultats sur les requêtes à mots-clés d'un site métier. Un modèle d'embedding coûterait
53 Mo de plus, soit un quart du budget.

---

## 5. Qualité attendue : être honnête

Gemma 3 270M est un généraliste de 270 M de paramètres. Il est bon pour sa taille, pas magique.

| Sait faire | Ne sait pas faire |
|---|---|
| Questions simples et courtes | Raisonnement en plusieurs étapes |
| Reformuler, résumer un court texte | Produire un texte long cohérent |
| Traduire des phrases simples (140+ langues) | Garantir des faits sans RAG |
| Suivre un format (JSON, liste, tableau) | Calcul fiable |
| Conversation de 2–3 tours | Mémoire d'un long historique |
| Dire « je ne sais pas » (si entraîné) | Être un ChatGPT |

**Le produit doit être conçu autour de ça :** RAG obligatoire pour le factuel, réponses courtes avec
sources, boutons plutôt que champ libre pour les cas fréquents, avertissement explicite, escalade
humaine quand le retrieval est faible.

---

## 6. Plan d'expériences

| # | Expérience | Critère go/no-go | Effort |
|---|---|---|---|
| **E0** | Produire le GGUF Q4_0 avec embeddings 4 bits et mesurer le RSS réel | fichier ≤ 155 Mo, RSS ≤ 200 Mo à 8 k | 1 j |
| **E1** | Qualité après quantification des embeddings | perplexité +5 % max, exactitude −3 pts max | 1 j |
| **E2** | Débit mono-flux et 4 slots sur le mini-PC cible | ≥ 8 tok/s/utilisateur | 1 j |
| **E3** | QLoRA français, 5–20 M tokens | +10 pts en win rate humain | 3 j |
| **E4** | RAG BM25 + garde-fous | ≥ 70 % sur la tâche | 3 j |
| **E5** | Comparaison à l'aveugle Gemma / MobileLLM / SmolLM2 | gagnant net sur 3 jeux/4 | 3 j |
| **E6** | Élagage du vocabulaire à 32 k (recherche) | perte ≤ 3 pts, fichier ≤ 110 Mo | 2 sem. |

**Ordre : E0 → E1 → E2 → E4 → E3 → E5 → E6.**
E0 et E1 se font en deux jours et décident de tout le reste. Si E1 échoue, on retombe sur
MobileLLM-R1-140M (170 Mo, mais spécialisé) ou sur un Q4_K_M à 253 Mo qui sort du budget.

---

## 7. Décisions

**ADR-006 — Cible d'inférence : 200 Mo de RSS.** 100 Mo reste l'aspiration.

**ADR-007 — Pas de pré-entraînement from scratch pour le produit.** QLoRA d'un modèle pré-entraîné ;
le from-scratch (60 M max sous 1 Go) devient un sujet de recherche, pas le chemin critique.

**ADR-008 — Déploiement llama-server, pool de contexte partagé, 8 k / 4 slots.**

**ADR-009 — La taille du vocabulaire est le levier mémoire n° 1.** 62 % des paramètres de
Gemma 3 270M sont des embeddings. Toute architecture cible utilise 32 k de vocabulaire.

**ADR-P-010 — Quantifier les embeddings en 4 bits est la condition de tenue sous 200 Mo.**
La QAT officielle le permet. Sans elle, le GGUF standard reste à 253 Mo.

**ADR-P-011 — Le budget 200 Mo est un working set résident** (apport de MNÉMOSYNE), pas une taille
de modèle. Toute proposition est jugée sur ses octets résidents.

---

## 8. Risques

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| La quantification des embeddings dégrade trop | moyenne | élevé | E1 ; repli sur Q4_K_M hors budget ou MobileLLM |
| Le débit sur N100 est insuffisant | moyenne | élevé | E2 avant achat ; viser Ryzen 5 |
| Les visiteurs attendent ChatGPT | élevée | élevé | cadrer le produit : RAG + réponses courtes |
| Fuite de prompt système | moyenne | moyen | filtre de sortie, tests adverses |
| Licence Gemma | certaine | moyen | vérifier les termes ; SmolLM2 en secours |

---

## 9. Ce qu'il faut retenir

1. **Un généraliste tient dans 200 Mo** : Gemma 3 270M QAT INT4 avec embeddings en 4 bits,
   183 Mo à 2 k de contexte, 193 Mo à 8 k.
2. **La condition est la quantification des embeddings**, pas le choix du modèle. Les GGUF
   « clés en main » les gardent à 8 bits et coûtent 100 Mo de trop.
3. **1 Go suffit pour adapter, pas pour pré-entraîner** : QLoRA, 5–20 M tokens, quelques heures,
   1–2 $.
4. **Le déploiement est bon marché** : un N100, 4 slots de 2 k, ~9 tok/s par visiteur, ~8 €/an.
5. **La qualité sera celle d'un petit assistant.** Le produit doit être bâti autour du RAG et de
   réponses courtes, pas présenté comme un ChatGPT.
6. **La seule recherche qui vaut le coup** est l'élagage du vocabulaire à 32 k (E6), qui ferait
   tomber le fichier à ~106 Mo et rendrait 32 k de contexte possible.

---

## Annexe — commandes

```bash
python tools/llm_budget.py table
python tools/llm_budget.py show gemma3-270m-q4emb-8k
python tools/llm_budget.py serve gemma3-270m-q4emb-8k --parallel 4 --ctx-per-slot 2048

# E0 : RSS réel sous plafond mémoire
systemd-run --scope -p MemoryMax=220M -p MemorySwapMax=0 \
  llama-server -m gemma-3-270m-qat-q4_0-emb4.gguf -c 8192 -np 4 -t 4 --metrics
grep -E 'VmRSS|VmHWM|Swap' /proc/$(pgrep -f llama-server)/status
```
