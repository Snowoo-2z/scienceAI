# MNÉMOSYNE — Le working set comme variable de conception
## ALEXANDRIE : un grand modèle dans ~100 Mo, sans grand transformer

**Agent** : MNÉMOSYNE · **Branche** : `arena/ab1c7e6e-scienceai` · **Date** : 2026-10-19
**Statut** : programme de recherche ouvert. Chaque chiffre est étiqueté
`MESURÉ`, `CALCULÉ` ou `HYPOTHÈSE`.
**Ne remplace pas** : l'onglet 12 (Gemma 3 270M QAT q4, 177 Mo) reste la
baseline produit. Ce document ouvre une branche de recherche, il ne décide
pas du produit.

---

## 0. En une page

Le projet a jusqu'ici optimisé le **nombre de paramètres**. C'est la mauvaise
variable.

La variable qui décide de la RAM est le **working set** : le nombre d'octets
qui doivent être résidents pour produire le prochain token. Un modèle peut
stocker 10 Md de paramètres sur disque et n'en garder que 34 Mo en RAM, si
seuls quelques experts sont actifs et si les experts partagent une base. Le
reste est du **stockage**, pas de la **mémoire**.

ALEXANDRIE :

- un **cœur** de 40 à 60 M de paramètres (25 à 35 Mo en 4 bits), distillé,
  qui tient le langage, le routage et la composition ;
- une **bibliothèque** de cartouches de compétence (0,1 à 4 Mo chacune) sur
  disque, formées de deltas de rang faible (LoRA par cartouche, ou diagonale
  dans une base partagée) ;
- un **routeur** qui charge 1 à 3 cartouches par **requête** (pas par token) ;
- un **composeur** qui additionne les deltas (arithmétique de tâches) ;
- un **vérificateur** (grammaire, calcul, outils) qui rattrape le petit
  modèle ;
- un **index** de faits sur disque, séparé des poids.

Budget `CALCULÉ` : **69 Mo de RSS** pour un cœur de 60 M + une bibliothèque
de 10 Md de paramètres ternaires sur 2,5 Go de disque, **161 tok/s**
(calcul-borné), **19 J par réponse** de 200 tokens. Le chiffre qui manque est
la **qualité** de cette capacité : c'est l'objet des expériences.

Ce n'est pas une compression d'un transformer dense. C'est un changement
d'unité de capacité : on ne demande plus « combien de paramètres tiennent dans
100 Mo ? », mais « combien de compétences peut-on stocker sur disque et charger
dans un working set de 34 Mo ? ».

Premier test falsifiable : à RSS égal (~100 Mo), un cœur 60 M + bibliothèque
bat-il un dense 120 M q4 ? Si non, on abandonne. Si oui, on a une voie vers un
généraliste sous 100 Mo sans pré-entraîner un grand modèle.

---

## 1. Pourquoi un nouvel axe

### 1.1 Ce que le projet a déjà établi

- `Agent-Recherche` a montré qu'un généraliste dense sous 100 Mo n'existe pas,
  mais qu'un modèle spécialisé 60–85 M est faisable (onglets 01–06).
- La cible a été portée à ~200 Mo (ADR-006), avec Gemma 3 270M QAT q4 à
  177 Mo de RSS à 8 k de contexte (onglet 12).
- Sous 1 Go, on peut **adapter** un 270 M par QLoRA, pas le pré-entraîner
  (ADR-007).
- L'énergie est dominée par la plateforme, pas par les poids (ADR-004).
- `KAIROS` a montré que le régime ~100 Go a sa propre physique : le noyau
  ternaire devient le goulet, et la compression est un **outil**, pas un
  objectif (ADR-007 de sa branche).
- `Silex` a montré qu'on peut factoriser un opérateur en `W_i = U diag(c_i) V`
  sans matérialiser la matrice, avec 9 tests d'équivalence < 1e-12, et que
  « plus d'opérateurs logiques » ne veut pas dire « plus de paramètres
  indépendants » (S2, commit `fbc1812`).

### 1.2 Ce que personne n'a encore relié

Trois résultats sont dans le dépôt, mais ne sont pas encore combinés :

1. la **sparsité contextuelle** : à chaque token, seule une petite partie des
   paramètres compte (Deja Vu : jusqu'à 85 % de sparsité, 7× de réduction) ;
2. la **localité des neurones** : une minorité de neurones « chauds » sert
   presque toujours (PowerInfer : 13,20 tok/s sur RTX 4090, +11,69× vs
   llama.cpp) ;
3. le **stockage flash** : on peut garder les poids hors DRAM et les charger à
   la demande (LLM in a flash : modèles 2× la DRAM, 4–5× CPU et 20–25× GPU).

La conclusion naturelle : si seuls quelques experts comptent par requête, et
si le reste peut vivre sur disque, alors **la taille du modèle n'est plus
bornée par la RAM**. Elle est bornée par le **working set** et par la
**qualité du routage**.

C'est exactement l'inverse de la démarche « réduire le modèle pour qu'il
rentre ». On garde une grande capacité, on la déplace, et on ne paie que ce
qu'on utilise.

### 1.3 La question que l'on pose

> À budget de RAM fixé, vaut-il mieux un modèle dense de 120 M, ou un cœur de
> 60 M capable de charger une bibliothèque de compétences de plusieurs
> milliards de paramètres ?

La réponse n'est pas évidente :

- le dense a plus de capacité **résidente** et une meilleure langue générale ;
- le modulaire a plus de capacité **spécialisée** et peut couvrir 20 à
  100 tâches sans augmenter le RSS ;
- le modulaire paie le routage, le chargement et l'interférence des deltas ;
- le dense paie la taille : à 100 Mo, il plafonne vers 120 M en q4.

Le projet doit trancher par l'expérience, pas par l'intuition.

---

## 2. La loi du working set

### 2.1 Définitions

| Symbole | Sens |
|---|---|
| `W` | budget de RSS du processus, en octets |
| `N_total` | paramètres stockés dans la bibliothèque (disque) |
| `N_res` | paramètres résidents (cœur + cartouches actives) |
| `N_act` | paramètres effectivement calculés par token |
| `b` | bits effectifs par poids, échelles incluses |
| `KV` | octets du KV cache / état récurrent |
| `R` | runtime, buffers, allocations fixes |
| `C` | cache de cartouches et de passages |
| `B_q` | octets lus sur disque pour charger les cartouches d'une requête |
| `BW` | bande passante disque utile, en octets/s |
| `GOPS` | débit de calcul utile, en opérations/s |

### 2.2 Les équations

```
RSS        ≈ N_res · b/8 + KV + R + C
tok/s      ≈ min( GOPS / (2 · N_act),  BW / B_token )
load_ms    ≈ B_q / BW · 1000
J/réponse  ≈ P_plateforme / tok/s_effectif · N_tokens
```

`B_token` est nul après le chargement si les cartouches sont chargées **par
requête**. C'est le régime que nous proposons : le disque paie une fois, le
calcul paie à chaque token.

Le seuil calcul/mémoire, généralisé de KAIROS :

```
GOPS ≥ 16 · BW / b
```

En dessous, le noyau de calcul est le goulet ; au-dessus, la mémoire (ou le
disque) l'est.

### 2.3 Le tableau de budget (`CALCULÉ`)

Configurations calculées par `tools/working_set.py` (Mo décimaux ; KV 10 Mo,
runtime 20 Mo, cache 5 Mo sauf mention) :

| Configuration | RSS | Poids actifs | Bibliothèque disque | tok/s | J/rép |
|---|---:|---:|---:|---:|---:|
| dense 60 M q4 | 68,8 | 33,8 | 67,5 | 166,7 | 18 |
| dense 120 M q4 | 99,5 | 67,5 | 135,0 | 83,3 | 36 |
| Gemma 3 270M QAT q4, 8 k | 176,5 | 134,0 | 268,0 | 74,6 | 40 |
| ALEX 60 M + cartouche 0,5 M | 69,0 | 34,0 | 34,0 | 165,3 | 18 |
| ALEX 60 M + bibliothèque 300 M ternaire | 69,2 | 34,2 | 108,8 | 161,3 | 19 |
| ALEX 60 M + bibliothèque 1 Md ternaire | 69,2 | 34,2 | 283,8 | 161,3 | 19 |
| ALEX 60 M + bibliothèque 10 Md ternaire | 69,2 | 34,2 | 2 533,8 | 161,3 | 19 |
| ALEX 100 M + bibliothèque 1 Md | 92,2 | 57,2 | 306,2 | 96,2 | 31 |

Hypothèses : `GOPS = 20` (2 vCPU, noyau naïf), plateforme 15 W, réponse de
200 tokens, chargement d'une cartouche de 2 Mo à 1 Go/s. Le dense 120 M est
le **vrai concurrent** : il tient tout juste sous 100 Mo, sans bibliothèque.

Lecture :

- à 60 M actifs, on est **calcul-borné** (166 tok/s) et le disque ne coûte
  que quelques millisecondes par requête ;
- la bibliothèque peut grossir de 300 M à 10 Md de paramètres **sans changer
  le RSS** ; seule la taille du disque change ;
- le 100 M actif reste sous 100 Mo mais coûte 96 tok/s : il faut choisir
  entre capacité résidente et débit ;
- le vrai saut serait de rendre le **cœur** plus capable sans augmenter son
  working set : c'est le rôle de la distillation et de la base partagée.

### 2.4 Ce que la loi ne dit pas

Elle ne dit **rien** de la qualité. Un bibliothèque de 10 Md de paramètres
mal routés vaut moins qu'un dense de 60 M. Les expériences de la section 6
sont là pour ça.

---

## 3. ALEXANDRIE

### 3.1 Vue d'ensemble

```
                 requête
                    │
                    ▼
        ┌───────────────────────┐
        │  ROUTEUR (1–5 M)      │  embedding + classifieur
        │  top-1 à top-3       │  confiance, fallback
        └──────────┬────────────┘
                   │  identifiants de cartouches
                   ▼
        ┌───────────────────────┐
        │  CHARGEUR             │  mmap / pread, LRU, prefetch
        │  cartouches 0,1–4 Mo  │  disque → RAM
        └──────────┬────────────┘
                   │  deltas de rang faible
                   ▼
        ┌───────────────────────┐
        │  CŒUR 40–60 M         │  base partagée U, V
        │  + Σ α_k Δ_k          │  composition par addition
        └──────────┬────────────┘
                   │  tokens
                   ▼
        ┌───────────────────────┐
        │  VÉRIFICATEUR         │  grammaire, calcul, outils
        │  + RETRIEVAL          │  faits, sources
        └───────────────────────┘
```

### 3.2 Le cœur

- 40 à 60 M de paramètres, 4 bits, ~25 à 35 Mo.
- Architecture : transformer compact (20 couches, `d=512`, GQA 8/2, FFN
  1 408, vocab 32 k, contexte 2 048) **ou** opérateurs partagés de Silex
  (`W_i = U diag(c_i) V`) pour réduire encore le stockage.
- Rôle : langue, format, refus, routage, composition. Pas les faits.
- Entraînement : distillation d'un enseignant 270 M à 8 B, 5 à 20 M tokens,
  QLoRA NF4, < 1 Go (voir §5).

### 3.3 Les cartouches

Une cartouche est un **delta de compétence**. Deux formats, à comparer :

```
Mode B — LoRA par cartouche (le défaut) :
    Δ_k = U_k V_k^T         2 r d paramètres par couche, U_k et V_k propres
Mode A — base partagée (compression extrême) :
    Δ_k = U diag(c_k) V     seulement r coefficients par couche
```

Le jouet E-M2-lite (§6.1bis) montre que le Mode A ne représente pas une
compétence arbitraire : il faut que la compétence soit déjà de rang faible
**dans la base partagée**. Le Mode B est donc le défaut ; le Mode A est un
pari à valider sur de vrais poids.

Ordres de grandeur (`CALCULÉ`) :

- LoRA r=8 sur 20 couches, `d=512` : `2 · 8 · 512 · 20 ≈ 164 k` paramètres,
  soit ~0,1 Mo en 4 bits ;
- LoRA r=32 : ~0,65 M paramètres, ~0,3 Mo ;
- diagonale partagée `c_k` de rang 256 : 256 valeurs par couche, ~1,3 ko par
  couche, ~26 ko pour 20 couches — **la cartouche la plus petite** ;
- une cartouche de 2 Mo peut contenir un LoRA de rang ~100 ou plusieurs
  têtes de compétence.

La bibliothèque peut donc contenir des **milliers** de cartouches sur
quelques Go. C'est le point clé : le stockage est abondant, la RAM ne l'est
pas.

### 3.4 Le routeur

- Entrée : embedding de la requête (ou des 64 premiers tokens).
- Sortie : 1 à 3 cartouches + un score de confiance.
- Taille : 1 à 5 M de paramètres (MLP sur embedding), ou un simple kNN sur
  des prototypes de compétence.
- Entraînement : labels « quelle cartouche améliore la perte sur cet
  exemple ? » — pas besoin d'un LLM juge.
- Sécurité : si la confiance est basse, on ne charge aucune cartouche et on
  répond avec le cœur + retrieval ; si elle est très basse, on escalade.
- Objectif : top-1 ≥ 90 %, top-3 ≥ 98 % sur le jeu de validation.

### 3.5 Le composeur

Deux modes, à comparer :

1. **Arithmétique de tâches** : `c = c_0 + Σ α_k c_k` (ou `W = W_0 + Σ α_k Δ_k`).
   Zéro paramètre supplémentaire, zéro temps de calcul supplémentaire.
   Risque : interférence entre deltas.
2. **Mélange gaté** : un petit gate apprend `α_k` par requête ; coût
   négligeable, mais il faut l'entraîner.

Le mode 1 est le pari scientifique : si les compétences sont des directions
presque orthogonales dans l'espace des poids, elles s'additionnent. La
littérature sur l'arithmétique de tâches le montre pour des tâches de vision
et de NLP ; il faut le vérifier à petite échelle, avec la base partagée de
Silex.

### 3.6 Le vérificateur et le retrieval

- **Grammaire** : décodage contraint (JSON, listes, formats) ; le petit modèle
  n'a pas à apprendre la syntaxe exacte.
- **Outils** : calculatrice, recherche, SQL, code ; le modèle propose, l'outil
  dispose.
- **Retrieval** : les faits vont dans un index (BM25 ou embeddings) sur
  disque ; le cœur lit les 3 à 5 passages pertinents. Le retrieval est le
  levier le plus rentable déjà documenté (onglet 12, E3).

### 3.7 La mémoire, niveau par niveau

| Niveau | Contenu | Taille typique | Support |
|---|---|---:|---|
| L0 | cœur, base partagée, routeur | 25–35 Mo | RAM |
| L1 | cartouches actives | 0,1–4 Mo | RAM (chargées) |
| L2 | KV cache / état | 5–15 Mo | RAM |
| L3 | index de faits, cache de passages | 5–20 Mo | RAM |
| L4 | bibliothèque complète | 0,1–10 Go | disque |
| L5 | index complet, corpus | 1–100 Go | disque |

Le budget 100 Mo porte sur L0–L3. L4–L5 sont du stockage. C'est la
définition explicite que nous proposons (ADR-009).

---

## 4. Ce qui est nouveau, et ce qui ne l'est pas

### 4.1 Prior art, cité honnêtement

- **MoE / sparsité** : Shazeer et al. 2017 ; Switch Transformer 2021 ;
  OLMoE 2024 (7 Md de paramètres, 1,3 Md actifs) ; DeepSeek-V3 ; Qwen3-Next.
- **Sparsité contextuelle** : Deja Vu (Liu et al., ICML 2023), jusqu'à 85 %
  de sparsité et 7× de réduction.
- **Localité des neurones** : PowerInfer (Song et al., 2023), 13,20 tok/s sur
  RTX 4090, +11,69× vs llama.cpp ; PowerInfer-2 (2024), 11,68 tok/s pour un
  Mixtral-47B sur smartphone.
- **Poids sur flash** : LLM in a flash (Alizadeh et al., ACL 2024), modèles
  2× la DRAM, 4–5× CPU et 20–25× GPU.
- **Adaptateurs** : LoRA (Hu et al., 2021), adapters, mixture of LoRAs.
- **Arithmétique de tâches** : Ilharco et al., ICLR 2023 ; les task vectors
  s'additionnent et se nient.
- **Retrieval** : RETRO (Borgeaud et al., 2022), kNN-LM, RAG.
- **Ternaire** : BitNet b1.58 (Ma et al., 2024), 2B4T (2025).
- **Budget** : KAIROS, loi `tok/s = BW/(B·ρ)` et seuil du noyau ; Silex,
  opérateurs partagés ; Agent-Recherche, budget 100/200 Mo.

### 4.2 Ce que nous ajoutons

1. **La loi du working set** comme critère de conception : `RSS` et `tok/s`
   en fonction de `N_res`, `N_act`, `B_q`, `BW`, et non du seul nombre de
   paramètres. Elle relie les régimes 100 Mo et 100 Go de KAIROS.
2. **La cartouche comme unité de capacité** : une compétence = un delta de
   rang faible dans une base partagée, stocké sur disque, chargé par requête.
   Le modèle peut grandir sans que le RSS grandisse.
3. **Le protocole d'entraînement < 1 Go** qui va avec : cœur distillé, une
   cartouche par compétence, routeur étiqueté par la perte, composition par
   arithmétique. Pas de pré-entraînement from scratch.
4. **Le test falsifiable** : à RSS égal, modular vs dense 120 M. Personne ne
   l'a chiffré dans ce projet.
5. **La comptabilité explicite** : RSS processus, page cache, disque, et
   énergie par réponse. Le débat « 100 Mo » ne peut pas être tranché sans
   cette comptabilité.

### 4.3 Ce que nous ne revendiquons pas

- Nous ne revendiquons pas 10 Md de paramètres **de qualité**. Nous
  revendiquons 10 Md de paramètres **stockés**, dont 2 M actifs. La qualité
  est l'expérience E-M2.
- Nous ne revendiquons pas une nouvelle architecture de transformer. La
  paramétrisation partagée est celle de Silex ; la nouveauté est le système
  et la loi.
- Nous ne revendiquons pas que le page cache est gratuit. S'il compte dans
  les 100 Mo, la bibliothèque doit tenir dans le cache ; c'est une question
  à trancher (voir §7).
- Nous ne revendiquons pas que les mesures du bac à sable se transfèrent au
  mini-PC. Les ratios se transfèrent, les valeurs absolues non.

---

## 5. Entraîner sous 1 Go

### 5.1 Vue d'ensemble

| Étape | Objet | Données | Mémoire | Durée |
|---|---|---|---:|---:|
| 1 | cœur 60 M, distillé | 5–20 M tokens | < 1 Go | heures GPU |
| 2 | base partagée `U, V` | initialisée par SVD du dense | < 1 Go | minutes |
| 3 | 20 cartouches | 1–5 M tokens chacune | < 1 Go | minutes-heures |
| 4 | routeur | 50–200 k exemples étiquetés | < 1 Go | minutes |
| 5 | composeur | paires/triplets | < 1 Go | minutes |
| 6 | évaluation | 20 × 200 exemples | — | heures |

Aucune étape ne dépasse 1 Go. Aucune étape ne pré-entraîne un modèle de
zéro. On reste dans l'esprit d'ADR-007.

### 5.2 Étape 1 — le cœur

Deux voies :

- **Distillation directe** : enseignant 270 M à 8 B, élève 60 M, 5 à 20 M
  tokens, QLoRA NF4 (r=16, α=32, seq 1 024, LR 1e-4), gradient
  checkpointing. Coût : quelques heures de GPU, 1 à 2 $.
- **Compression SVD** : partir de Gemma 3 270M, décomposer chaque matrice
  `W ≈ U diag(c) V` à rang `r`, initialiser le cœur avec `U, V` et `c`,
  puis distiller brièvement pour récupérer la perte. C'est la voie la plus
  intéressante : elle ne part pas de zéro et elle teste directement la
  paramétrisation partagée.

### 5.3 Étape 2 — la base partagée

Si l'on utilise la paramétrisation de Silex :

```
W_i = U diag(c_i) V
```

- `U` et `V` sont partagés par toutes les couches/compétences ;
- `c_i` est la cartouche (ou une partie de la cartouche) ;
- le rang `r` contrôle la capacité ;
- l'initialisation par SVD du dense donne un point de départ raisonnable ;
- la question ouverte est le **rang nécessaire** pour ne pas perdre plus de
  5 % de perplexité. C'est E-M1.

**Avertissement du jouet E-M2-lite** : une base `U, V` fixe ne représente
pas une compétence arbitraire (erreur ~0,98 sur une matrice dense aléatoire,
même à `r = d`). Le Mode A n'est viable que si les compétences réelles sont
déjà alignées avec la base. Tant que E-M1 ne l'a pas montré, le défaut est
le **Mode B** (facteurs `U_k, V_k` propres à chaque cartouche).

### 5.4 Étape 3 — les cartouches

Pour chaque compétence :

1. constituer 1 à 5 M tokens d'instructions/exemples de la compétence ;
2. geler le cœur ;
3. entraîner un LoRA (ou une diagonale `c_k`) sur les couches FFN et
   attention ;
4. évaluer sur un jeu de 200 exemples ;
5. stocker le delta + un prototype d'embedding pour le routeur.

La cartouche n'a pas besoin de contenir la langue : le cœur la fournit. Elle
contient le **style**, le **format**, le **vocabulaire de domaine**, les
**règles** de la compétence.

### 5.5 Étape 4 — le routeur

- Label : pour chaque exemple, on teste les 20 cartouches (ou un sous-ensemble)
  et on garde celle qui minimise la perte.
- Entrée : embedding moyen de la requête (petit modèle de 1–5 M, ou
  sentence-transformers gelé).
- Sortie : distribution sur les cartouches + « aucune ».
- Calibration : température, seuil de confiance, fallback.
- Coût : négligeable.

### 5.6 Étape 5 — la composition

- **Mode arithmétique** : `c = c_0 + α_1 c_1 + α_2 c_2` ; `α` fixé à 1 ou
  appris sur un petit jeu de paires.
- **Mode gaté** : un MLP de 100 k paramètres prend l'embedding et sort les
  `α`.
- **Évaluation** : tâches composées (traduire + format JSON, résumer +
  domaine, etc.). Si l'interférence dépasse 5 points, on route une seule
  cartouche à la fois.

---

## 6. Expériences et falsification

### 6.1 E-M0 — chargement des cartouches (`MESURÉ`, bac à sable)

Protocole : `tools/wsprobe.c`, fichier de 512 Mo et 2 Go, `pread`,
`posix_fadvise(DONTNEED)` + `drop_caches`, 16 tirages par taille, médiane et
moyenne. Matériel : Xeon 2,60 GHz, 2 vCPU, 4 Go de RAM, ext4, 2026-10-19.

Résultats (médianes, lecture bufferisée sur fichier de 2 Go) :

| Taille | Médiane | Moyenne | Min | Max |
|---:|---:|---:|---:|---:|
| 256 Ko | 0,34 ms | 0,36 ms | 0,27 ms | 0,51 ms |
| 512 Ko | 0,72 ms | 0,70 ms | 0,50 ms | 0,94 ms |
| 1 Mo | 1,01 ms | 1,03 ms | 0,95 ms | 1,16 ms |
| 2 Mo | 1,86 ms | 8,40 ms | 1,30 ms | 101,76 ms |
| 4 Mo | 3,07 ms | 9,18 ms | 2,27 ms | 103,67 ms |
| 8 Mo | 5,75 ms | 12,14 ms | 5,09 ms | 107,45 ms |

En `O_DIRECT` (512 Mo) : 1 Mo = 0,34 ms, 2 Mo = 0,61 ms, 4 Mo = 0,95 ms,
8 Mo = 2,45 ms. Débit séquentiel : 0,8 à 3,4 Go/s selon la taille du fichier
et l'état du cache.

Lecture : charger une cartouche de 2 à 4 Mo coûte **1 à 4 ms en médiane**,
soit moins de 0,2 % d'une réponse de 200 tokens à 10 tok/s. Même avec des
p99 à 100 ms (VM partagée), le chargement par requête reste négligeable. En
revanche, un pagination **par token** d'un cœur de 34 Mo à 500 Mo/s coûterait
68 ms/token (15 tok/s) : c'est le régime à éviter. D'où le choix
« cartouche par requête, cœur résident ».

Les valeurs absolues ne se transfèrent pas au mini-PC : il faut refaire E-M0
sur le matériel cible (SSD SATA/NVMe, page cache, charge). Le protocole et
le code, eux, se transfèrent.

### 6.1bis E-M2-lite — capacité et composition de la base partagée (`MESURÉ`, jouet)

Protocole : `experiments/mnemosyne/toy_shared_basis.py`, `d=32`, 5 graines,
rangs 1 à 32. Ce n'est **pas** un LLM : c'est un test de ce que la
paramétrisation `W_k = U diag(c_k) V` peut représenter, à base `U, V`
partagée et fixée.

Erreur relative de Frobenius (moyenne ± écart-type) :

| `r` | matrice dense aléatoire | dans la base | dans la base + 10 % de bruit | autre base |
|---:|---:|---:|---:|---:|
| 1 | 1,000 ± 0,000 | 0,000 ± 0,000 | 0,682 ± 0,295 | 0,999 ± 0,001 |
| 2 | 0,999 ± 0,001 | 0,000 ± 0,000 | 0,518 ± 0,300 | 1,000 ± 0,000 |
| 4 | 0,998 ± 0,001 | 0,000 ± 0,000 | 0,327 ± 0,118 | 0,998 ± 0,001 |
| 8 | 0,996 ± 0,002 | 0,000 ± 0,000 | 0,287 ± 0,136 | 0,997 ± 0,001 |
| 16 | 0,987 ± 0,002 | 0,000 ± 0,000 | 0,131 ± 0,026 | 0,993 ± 0,003 |
| 32 | 0,982 ± 0,004 | 0,000 ± 0,000 | 0,089 ± 0,012 | 0,984 ± 0,004 |

Composition `c_A + c_B` contre le meilleur ajustement direct de `W_A + W_B` :

| `r` | addition | ajustement direct | écart |
|---:|---:|---:|---:|
| 1 | 0,660 ± 0,248 | 0,659 ± 0,248 | 0,0004 |
| 8 | 0,135 ± 0,020 | 0,135 ± 0,019 | 0,0008 |
| 32 | 0,074 ± 0,012 | 0,073 ± 0,012 | 0,0012 |

Stockage (20 compétences) : le partage coûte 84 à 2 688 paramètres, contre
20 480 pour 20 matrices denses de 32×32, soit un ratio de 0,004 à 0,131.

Lecture :

1. **Une base partagée fixe n'a que `r` degrés de liberté par compétence.**
   Même à `r = d = 32`, elle ne représente pas une matrice dense aléatoire
   (erreur ~0,98) : la famille est `U diag(c) V`, pas « toutes les matrices
   de rang r ». C'est beaucoup plus restrictif que LoRA, qui apprend `U_k`
   et `V_k` par compétence.
2. **Si la compétence est dans la base, le fit est exact** (erreur 0,000).
   Avec 10 % de bruit hors base, l'erreur est 0,089 à `r=32`, 0,287 à `r=8`,
   0,682 à `r=1`. L'erreur est l'énergie hors base.
3. **Une compétence apprise dans une autre base ne se transfère pas**
   (erreur ~1,0). La base doit être partagée par construction, pas apprise
   indépendamment.
4. **La composition par addition est optimale dans la base** : `c_A + c_B`
   égale le meilleur ajustement direct à moins de 0,001. Si les compétences
   sont dans la base, la composition est gratuite et sans interférence.
   Le problème est la **représentation**, pas la composition.
5. Le partage est 7,6× à 244× plus compact que 20 matrices denses, selon
   `r`.

**Conséquence pour ALEXANDRIE** : par défaut, utiliser des **cartouches
LoRA par compétence** (Mode B : `U_k, V_k, c_k`, 0,1 à 0,3 Mo), qui peuvent
représenter n'importe quelle compétence de rang faible. La **base partagée**
(Mode A : `c_k` seul, quelques dizaines de ko) reste un pari de compression
extrême, à valider sur de vrais poids par E-M1. Le jouet ne contredit pas
Silex S2 (mécanique d'un opérateur) ; il montre la **limite de capacité**
d'un `U, V` partagé entre compétences. Silex S3 doit mesurer la courbe
rang-capacité sur de vrais poids, pas sur des matrices aléatoires.

### 6.2 E-M1 — rang nécessaire de la base partagée (critique)

**Question** : à quel rang `r` la paramétrisation `W = U diag(c) V` perd-elle
moins de 5 % de perplexité par rapport au dense, à taille égale ?

Protocole :

1. prendre Gemma 3 270M ou un petit modèle dense 60–120 M ;
2. pour chaque couche, SVD tronquée à `r ∈ {16, 32, 64, 128, 256}` ;
3. initialiser `U, V, c` ;
4. distiller 1 à 5 M tokens pour récupérer ;
5. mesurer la perplexité et les benchmarks de la tâche cible.

Critère go/no-go : `r ≤ 25 % de d` avec < 5 % de perte. Sinon, la base
partagée est trop restrictive et ALEXANDRIE perd son intérêt.

Collaboration : Silex a déjà le prototype S2 et le protocole S3 ; cette
expérience est la version « vrai modèle » de S3.

### 6.3 E-M2 — modular vs dense à RSS égal (le juge de paix)

**Question** : à ~100 Mo de RSS, un cœur 60 M + bibliothèque bat-il un dense
120 M q4 ?

Protocole :

1. 20 compétences, 200 exemples chacune (tâches du site cible) ;
2. dense 120 M q4, entraîné/distillé sur le même total de tokens ;
3. cœur 60 M + 20 cartouches, routeur top-1 ;
4. même budget de tokens d'entraînement, même matériel, même évaluation ;
5. mesurer : exactitude moyenne, pire compétence, taux de routage,
   latence, RSS, J/réponse.

Critère go/no-go : modular ≥ dense + 5 points en moyenne **et** + 15 points
sur les compétences spécialisées, à ≤ 2× la latence. Si non, on abandonne
la bibliothèque et on garde le dense.

### 6.4 E-M3 — routage

- top-1 ≥ 90 %, top-3 ≥ 98 % ;
- si non : routeur plus grand (5–10 M), retrieval de prototypes, ou
  routage par perte de reconstruction ;
- mesurer le coût des erreurs : une mauvaise cartouche est-elle pire que pas
  de cartouche ?

### 6.5 E-M4 — oubli catastrophique

- ajouter 10 cartouches séquentiellement ;
- mesurer la rétention sur les 10 premières ;
- critère : < 5 % de baisse ;
- si non : entraîner les cartouches avec un terme de orthogonalité, ou
  router une seule cartouche.

### 6.6 E-M5 — prototype système

- implémenter le chargeur + routeur en Python (simulation) puis en C/llama.cpp
  (si le temps le permet) ;
- mesurer RSS, latence, débit sur le bac à sable et sur un mini-PC ;
- vérifier que le RSS mesuré correspond au calcul à ± 10 %.

### 6.7 E-M6 — déploiement client (option)

- modèle 60 M q4 (~30 Mo) dans le navigateur (WebGPU/WASM) ;
- serveur : fichiers statiques + cache, RSS ~0 ;
- mesurer téléchargement, RAM du navigateur, latence ;
- intérêt : le serveur n'infère pas, la vie privée est meilleure, le coût
  serveur est nul. Limite : 30 Mo de téléchargement et la RAM du navigateur
  ne sont pas 100 Mo.

---

## 7. Ce qui falsifierait ALEXANDRIE

Écrit à l'avance, pour éviter de déplacer les critères après les résultats :

1. **E-M1** : rang > 25 % de `d` nécessaire pour < 5 % de perte → la base
   partagée est trop restrictive.
2. **E-M2** : modular < dense 120 M à RSS égal → la bibliothèque ne vaut pas
   son coût.
3. **E-M3** : routage top-1 < 80 % → le système charge la mauvaise
   compétence ; il faut un routeur plus cher, ce qui annule le gain.
4. **E-M4** : oubli > 10 % → la composition/additivité est illusoire.
5. **E-M0** : chargement > 100 ms p99 sur le matériel cible → il faut
   précharger ou réduire les cartouches.
6. **Comptabilité** : si le page cache compte dans les 100 Mo et que la
   bibliothèque ne tient pas, le régime « disque » disparaît ; il reste le
   régime « petites cartouches en RAM ».

Si l'un de ces critères tombe, on le dit et on change de branche. Un résultat
négatif est un livrable.

---

## 8. Collaboration

### 8.1 À KAIROS

- Ta loi `tok/s = BW/(B·ρ)` et ton seuil `GOPS ≥ 16·BW/bits` sont la base de
  la mienne. Je les généralise à une hiérarchie RAM/disque avec un taux de
  hit et un chargement par requête.
- Ton régime ~100 Go est la **borne haute** de la même loi : chez toi, le
  disque n'existe pas, tout est en RAM, et le noyau est le goulet. Chez moi,
  le disque existe, le cœur est petit, et le routage est le goulet.
- Question : peux-tu tester le noyau de chargement de cartouches (pread
  aléatoire 1–8 Mo) sur ta machine cible, et me donner le p99, pas seulement
  la moyenne ? C'est E-M0 côté matériel.
- Ton ADR-007 (« la compression est un outil, la métrique est le rendement
  marginal ») est exactement le cadre de ce document. Je propose de le
  compléter par « le rendement marginal se mesure par octet de **working
  set**, pas par octet de fichier ».

### 8.2 À Silex

- Ton opérateur `W_i = U diag(c_i) V` est le format de cartouche idéal :
  la base `U, V` reste en RAM, la diagonale `c_i` est la compétence, et
  l'addition de deux diagonales est une composition.
- Ton résultat négatif (« plus d'opérateurs ≠ plus de paramètres
  indépendants ») est le bon garde-fou : il faut mesurer la capacité
  **apprise**, pas le nombre d'opérateurs.
- Proposition : co-écrire S3 comme E-M1/E-M2. Tu fournis le prototype et la
  mesure d'équivalence ; j'apporte le budget, le routeur et le jeu de
  compétences. On publie les deux résultats, positifs ou négatifs.

### 8.3 À Agent-Recherche

- Je ne touche pas à l'onglet 12 ni au produit Gemma 270M. Ta baseline est
  le point de comparaison à 200 Mo.
- J'ai besoin de : (a) la config exacte du GGUF Q4_0 et le RSS mesuré en E0 ;
  (b) le jeu d'évaluation de la tâche cible (200–500 exemples) ; (c) le
  budget de tokens utilisé pour le QLoRA.
- Si E-M2 échoue, je le dirai ; si E-M2 réussit, on aura une voie vers un
  généraliste à 100 Mo qui **complète** ton produit à 200 Mo.

### 8.4 À W2, W4, W5, W7

- **W2** : 20 compétences × 1–5 M tokens, plus les labels de routage. C'est
  le vrai goulot : sans données de compétence, la bibliothèque est vide.
- **W4** : le routeur, l'index de retrieval, le vérificateur, la grammaire.
- **W5** : la comptabilité RSS/page cache/disque et `J/réponse juste`.
- **W7** : le chargeur de cartouches dans le serveur web, le cache, la
  supervision.

### 8.5 Décision proposée

**ADR-009 — Le working set est la variable de conception.**

- Ouvrir le chantier W1-ter « ALEXANDRIE » sur la branche
  `arena/ab1c7e6e-scienceai`.
- Ne pas changer la baseline produit (Gemma 3 270M, 200 Mo).
- Ajouter au protocole de mesure les champs : RSS processus, page cache,
  octets lus sur disque, taille de la bibliothèque.
- Demander à l'humain : (a) les 100 Mo sont-ils le RSS du processus ou la
  RAM totale, page cache inclus ? (b) un disque de 1–10 Go est-il acceptable
  sur la cible ? (c) la cible est-elle un site web avec 20–100 tâches
  récurrentes, ou une conversation ouverte ? La réponse change le point de
  fonctionnement.

---

## 9. Questions à l'humain

1. **Unité** : les ~100 Mo sont-ils le RSS du processus d'inférence (hors
   page cache), ou la RAM totale de la machine, page cache inclus ? Le
   régime « bibliothèque sur disque » n'existe que dans le premier cas.
2. **Stockage** : la cible accepte-t-elle 1 à 10 Go de disque pour la
   bibliothèque ? Si non, la bibliothèque doit tenir en RAM, et le gain
   disparaît.
3. **Distribution des requêtes** : le site vise-t-il 20 à 100 tâches
   récurrentes (où la bibliothèque excelle) ou de la conversation ouverte
   (où le dense 270M reste meilleur) ?
4. **Latence** : quel est le budget par requête ? 5 s, 20 s, 60 s ? Le
   chargement des cartouches et le retrieval doivent tenir dedans.
5. **Qualité** : quel est le critère de succès ? Le projet propose
   « ≥ 70 % sur la tâche, ≥ 90 % de format, 0 fuite de prompt ». Le même
   critère doit s'appliquer au modular et au dense.

---

## 10. Références

### Internes au projet

- `Agent-Recherche`, `docs/RECHERCHE-LLM-100MB.md` et
  `docs/ARCHITECTURE-200MB-WEB.md`, branches `arena/9c2ae177-scienceai`.
- `KAIROS`, `HECATON.md`, `docs/01-LOI-DE-BUDGET.md`, `docs/RESUME.md`,
  branche `arena/c9070fe4-scienceai`.
- `Silex`, `docs/research/silex-streaming.md`, `experiments/silex/`,
  branche `arena/d930d4b4-scienceai`, commit `fbc1812`.
- `AGENTS.md`, `coordination/decisions.md`, onglets 11 et 12 du document
  partagé.

### Externes

1. K. Alizadeh et al., *LLM in a flash: Efficient Large Language Model
   Inference with Limited Memory*, arXiv:2312.11514, ACL 2024.
2. Y. Song et al., *PowerInfer: Fast Large Language Model Serving with a
   Consumer-grade GPU*, arXiv:2312.12456, 2023 ; *PowerInfer-2*, 2024.
3. Z. Liu et al., *Deja Vu: Contextual Sparsity for Efficient LLMs at
   Inference Time*, arXiv:2310.17157, ICML 2023.
4. N. Muennighoff et al., *OLMoE: Open Mixture-of-Experts Language Models*,
   arXiv:2409.02060, 2024.
5. G. Ilharco et al., *Editing Models with Task Arithmetic*,
   arXiv:2212.04089, ICLR 2023.
6. S. Ma et al., *The Era of 1-bit LLMs: All Large Language Models are in
   1.58 Bits*, arXiv:2402.17764, 2024 ; *BitNet b1.58 2B4T*, arXiv:2504.12285,
   2025.
7. E. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*,
   arXiv:2106.09685, ICLR 2022.
8. S. Borgeaud et al., *Improving Language Models by Retrieving from
   Trillions of Tokens* (RETRO), arXiv:2112.04426, ICML 2022.
9. Gemma Team, *Gemma 3 Technical Report*, 2025.
10. N. Shazeer et al., *Outrageously Large Neural Networks: The
    Sparsely-Gated Mixture-of-Experts Layer*, arXiv:1701.06538, ICLR 2017.

---

## Annexe A — Reproduire

```bash
# Budget
python3 tools/working_set.py table
python3 tools/working_set.py show alex-60m-lib10g
python3 tools/working_set.py law
python3 tools/working_set.py csv > results/working-set-configs.csv

# Chargement des cartouches
gcc -O3 -Wall -Wextra -o /tmp/wsprobe tools/wsprobe.c
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
/tmp/wsprobe --file /tmp/wsprobe.bin --size-mb 512 --n 64
/tmp/wsprobe --file /tmp/wsprobe2.bin --size-mb 2048 --n 32
/tmp/wsprobe --file /tmp/wsprobe3.bin --size-mb 512 --n 32 --direct
```

## Annexe B — Champs de mesure

Pour être comparable, tout chiffre de ce document doit préciser :

- **modèle / config** : nom, `N_total`, `N_act`, rang, bits ;
- **quantification** : 4 bits, 4,5 bits, 2 bits, 1,58 bit ;
- **contexte** : tokens de contexte, KV ;
- **backend** : `pread`, llama.cpp, PyTorch, etc. ;
- **matériel** : CPU, RAM, disque, OS ;
- **date** : 2026-10-19.

Les chiffres du §6.1 sont `MESURÉ` sur le bac à sable ; ceux du §2.3 sont
`CALCULÉ` ; les projections de qualité sont `HYPOTHÈSE`.

— MNÉMOSYNE
