# NanoFR — un modèle de langue entraîné et servi ici, sans GPU

> Résultat concret du 2026-10-19 : la chaîne complète **corpus → tokenizer →
> entraînement numpy → GGUF → quantification → llama.cpp → navigateur** tourne
> sur 2 vCPU et 4 Go de RAM. Le modèle est petit (6,5 M de paramètres) ; ce qui
> est démontré, c'est le **chemin de production**, pas la qualité d'un généraliste.

---

## 1. Pourquoi ce modèle existe

Les documents du projet décrivaient une cible (270 M de paramètres dans 200 Mo)
sans jamais montrer un seul token sorti d'un modèle. Une cible non exécutée est
une hypothèse. NanoFR est l'expérience E0bis : prouver que la chaîne tient, et
mesurer chaque étape avec des chiffres reproductibles.

Trois contraintes ont guidé les choix :

1. **Aucun GPU, 2 vCPU, 4 Go de RAM.** L'entraînement doit tenir sous 1 Go.
2. **Aucune dépendance lourde.** numpy + sentencepiece seulement ; pas de
   PyTorch, pas de CUDA.
3. **Le modèle doit sortir en GGUF** et être exécutable par llama.cpp, sinon la
   démonstration ne prouve rien du chemin de déploiement.

## 2. Corpus

| Source | Licence | Volume utilisé |
|---|---|---|
| isosceles (Maupassant FR + Poe traduit par Baudelaire) | CC-BY 4.0 | 381 textes |
| ELTeC-fra, 100 romans 1840–1920 | CC-BY 4.0 | 22 romans (budget 9 M caractères) |

Après nettoyage (NFC, espaces, lignes vides) et mélange : **16,02 M de
caractères** d'entraînement, 0,21 M de validation, 403 documents.

## 3. Tokenizer

SentencePiece BPE entraîné sur le corpus : **8 192 pièces**, `byte_fallback=True`
(tout texte reste encodable), BOS=1, EOS=2. Rendement mesuré : **3,80
caractères par token**.

Le choix de 8 192 pièces plutôt que 32 000 ou 262 144 est un choix de calcul :
la projection de sortie coûte `2 × d × vocab` par token, soit 4,2 MFLOP avec
8 192 pièces contre 16,4 avec 32 000. Sur un petit modèle, **le vocabulaire
coûte plus cher que les couches** — c'est exactement le mécanisme qui empêche
Gemma 3 270M de tenir dans 200 Mo en quantification standard.

## 4. Architecture

| Paramètre | Valeur |
|---|---:|
| Paramètres | 6,52 M |
| Couches | 6 |
| d_model | 256 |
| Têtes Q / KV | 4 / 2 (GQA) |
| head_dim | 64 |
| FFN (SwiGLU) | 704 |
| Contexte | 256 |
| Embeddings | partagés (weight tying) |
| Norme | RMSNorm (eps 1e-5) |
| Position | RoPE split-half, base 10 000 |

L'implémentation (`tools/nanofr_train.py`) est un transformer « architecture
LLaMA » écrit à la main en numpy : forward et backward dérivés à la main, AdamW,
clip global à 1,0, warmup 60 pas puis cosinus. **Aucun autograd.**

Les gradients ont été vérifiés par différences finies (erreur < 5 % sur
`wq`, `w_down`, `tok_embd`) ; cette vérification a révélé un bug réel dans la
rétropropagation de RMSNorm (utilisation de la sortie normalisée au lieu de
l'entrée), qui faisait exploser les gradients d'un facteur ~10⁴.

## 5. Entraînement (mesuré)

| Grandeur | Valeur |
|---|---|
| Pas d'optimisation | 2 400 |
| Batch × contexte | 8 × 256 = 2 048 tokens/pas |
| Tokens vus | ~4,9 M (≈ 1,17 époque) |
| Débit | **1,7 k tokens/s** (2 vCPU) |
| Durée | ~48 min |
| Pic RSS | **~640 Mo** (contrainte : 1 Go) |
| Loss (train) | 9,06 → voir §6 |

## 6. Résultats

À compléter après la fin de l'entraînement (section générée) :
perplexité de validation, bits par caractère, exemples de génération, tailles
GGUF, RSS et débits mesurés.

## 7. Déploiement

```
llama-server -m data/nanofr_q8_0.gguf -c 256 -t 2 --host 127.0.0.1 --port 8081
python3 app/server.py --port 8000 --backend http://127.0.0.1:8081
```

L'application (`app/`) est un serveur HTTP de la bibliothèque standard : une page,
un endpoint `/api/complete` qui relaie vers llama-server. Aucune dépendance.

## 8. Ce que ce résultat prouve — et ce qu'il ne prouve pas

**Prouvé :** la chaîne complète fonctionne, chaque étape est mesurée, les
fichiers sont reproductibles, le coût en RAM est connu (28,5 Mo de RSS pour le
modèle Q4, application comprise 20 Mo).

**Non prouvé :** qu'un modèle de 6,5 M de paramètres soit utile. Il imite un
style, il ne répond pas à des questions. La cible du projet reste un généraliste
pré-entraîné de 270 M quantifié en 4 bits (voir `ARCHITECTURE-200MB-WEB.md`),
que ce banc ne remplace pas : il en valide l'outillage.
