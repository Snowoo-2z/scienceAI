# Expériences, risques, idées à tester

**KAIROS · 2026-10-19**

Chaque expérience a un **critère d'échec explicite**. Une idée sans protocole
de réfutation n'entre pas dans cette liste.

---

## 1. Plan d'expériences

### E-K1 — Mesurer le plancher du runtime à 85 Go (MESURER)

**Question :** quel est le RSS réel de llama.cpp (ou bitnet.cpp) quand on
charge 80 Go de poids ? Le runtime est-il de 1,5 Go (hypothèse) ou de 4 Go ?

**Protocole :** charger un modèle MoE quantifié de 40 à 80 Go, mesurer le RSS
au pic avec `/usr/bin/time -v` et `smem -t`, à 2 k et 32 k de contexte.
Trois backends : llama.cpp CPU, bitnet.cpp, KTransformers.

**Coût :** 2 jours, machine 96-128 Go.

**Échec :** si le runtime dépasse 3 Go, la marge de 11 Go tombe à 9 ; si elle
devient négative, il faut retirer les couches mémoire (−4,6 Go) ou passer à
215 B.

### E-K2 — La taxe ternaire (LE VERROU)

**Question :** à budget mémoire constant, vaut-il mieux 4 bits / 180 B ou
1,7 bit / 372 B ?

**Protocole :** 4 modèles de 80 M à 1 B (même tokenizer, mêmes données),
quantifiés par QAT à 1,58 / 2 / 3 / 4 / 8 bits, 20 Md tokens chacun.
Mesurer la perte de validation + MMLU + GSM8K + TriviaQA. Ajuster le terme
`γ_bits` d'une loi de la forme `L = L∞ + A/P^α + B/D^β + γ·max(0, 4−b)²`.

**Coût :** 1 500 à 3 000 $, 2 semaines.

**Échec :** si à 1 B la perte 1,58 bit vs 4 bits dépasse 5 points, le ternaire
n'est pas viable à cette échelle → capacité max 180 B, débit ÷ 2.

**C'est l'expérience la plus importante du document.** Elle décide si le
projet achète 372 B ou 180 B.

### E-K3 — Le noyau ternaire réel (BLOQUANT)

**Question :** quel GOP/s obtient-on sur 16 cœurs AVX-512 avec un GEMV
ternaire optimisé ?

**Protocole :** partir de `tools/tgemv.c` (8,5 GOP/s sur 2 vCPU), implémenter
le déballage 2 bits + `vpdpbusd`, vectoriser sur 16 cœurs, mesurer sur les
géométries réelles (2 048 × 1 024 et 2 048 × 8 192).

**Coût :** 1 semaine, 0 €.

**Échec :** si < 400 GOP/s, le débit plafonne à 33 tok/s et la cible « ≥ 40
tok/s sur une machine à 3 500 € » est manquée.

### E-K4 — Les couches mémoire aident-elles ?

**Question :** 25 à 35 % des paramètres en couches produit-clé ajoutent-ils de
la qualité factuelle à budget de bande passante constant ?

**Protocole :** base 1 B, 4 variantes (0 / 15 / 30 / 45 % des FFN remplacés
par de la mémoire), 100 Md tokens, même budget de lecture par token.

**Coût :** ~20 000 $, 3 semaines.

**Échec :** < +2 points sur TriviaQA/NQ → retirer les couches mémoire,
−9,2 B de capacité et −4,6 Go de RAM.

### E-K5 — Le routeur de rang faible dégrade-t-il le routage ?

**Question :** un routeur `d → 256 → E` (0,05 Go/token) vaut-il un routeur
dense `d → E` (0,25 Go/token) ?

**Protocole :** même modèle, deux routeurs, mesurer l'entropie de routage, le
taux de charge des experts et la perte.

**Coût :** 3 jours, ~500 $.

**Échec :** si le taux d'experts morts dépasse 20 %, revenir au routeur dense
et accepter +10 % de lecture.

### E-K6 — Attention hybride 1/4 vs 1/2

**Question :** 16 couches d'attention pleine suffisent-elles pour le rappel
long contexte (32 k) ?

**Protocole :** abliter l'attention sur 16 vs 32 couches, mesurer RULER,
LongBench et la perte de validation.

**Coût :** 10 000 $, 2 semaines.

**Échec :** si RULER perd > 5 points à 32 k, remonter à 1/2 → KV ×2 (+0,15
Go/token, −7 % de débit).

### E-K7 — Le prefill sur iGPU

**Question :** peut-on descendre sous 2 s pour 1 000 tokens de prompt ?

**Protocole :** prefill CPU (15 s estimé) vs iGPU Radeon 8060S vs hybride
(couches d'attention sur iGPU, experts sur CPU).

**Coût :** 1 semaine, machine Strix Halo.

**Échec :** si > 5 s, le produit n'est pas interactif et il faut un GPU
dédié — ce qui change le budget matériel.

### E-K8 — Qualité de Qwen3-Next-80B en ternaire (chemin A)

**Question :** le meilleur modèle ouvert « 100 Go » perd combien en passant à
1,58 bit ?

**Protocole :** quantifier, réparer sur 100 Md tokens, évaluer sur MMLU,
GSM8K, HumanEval, MT-Bench.

**Coût :** 5 000 $, 3 semaines.

**Échec :** si > 10 points de perte, le ternaire QAT n'est pas une voie
rentable et il faut rester à 4 bits.

### E-K9 — J/reponse juste, HECATON vs nano

**Question :** combien de joules pour une réponse **correcte**, 372 B contre
75 M ?

**Protocole :** 500 questions, 3 domaines, mesure des joules au mur et du
taux de réussite. `J_correct = J_total / taux_de_réussite`.

**Coût :** 1 semaine, 0 € (si les modèles existent).

**Échec :** si le petit modèle gagne, HECATON est un projet de recherche, pas
un produit.

### E-K10 — Stabilité du routage après upcycling

**Question :** dupliquer 512 → 896 experts puis réparer produit-il des
experts réellement différents ?

**Protocole :** mesurer la divergence KL entre experts dupliqués avant/après
réparation, et le taux d'experts jamais sélectionnés.

**Coût :** 1 900 $, 1 semaine.

**Échec :** si la divergence reste < 5 %, l'upcycling n'ajoute pas de capacité
et le chemin B est mort.

---

## 2. Registre des risques

| # | Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Le ternaire coûte > 5 points à 300 B | moyenne | **fatal** | E-K2 avant tout investissement |
| R2 | Le noyau plafonne à 300 GOP/s | moyenne | grave (÷2 débit) | E-K3, budget kernel dédié |
| R3 | Les couches mémoire dégradent le raisonnement | moyenne | −9 B de capacité | E-K4, retrait possible |
| R4 | Impossible d'entraîner 372 B à ce budget | **certaine** | capacité réelle 215 B | chemin A, assumer 215 B |
| R5 | Le prefill CPU est inutilisable | élevée | produit non interactif | E-K7, iGPU ou GPU |
| R6 | Qwen3-Next ternaire s'effondre | faible-moyenne | chemin A mort | E-K8, repli 4 bits |
| R7 | La RAM réelle dépasse 96 Go | faible | besoin de 128 Go | marge de 11 Go, E-K1 |
| R8 | Les prix de la DDR5 montent encore | élevée | +20-40 % matériel | acheter d'occasion, 64 Go suffit pour 215 B |
| R9 | Personne ne mesure `J/réponse` | **certaine** | on optimise le mauvais chiffre | E-K9, W5 |
| R10 | Le projet se disperse entre 100 Mo et 100 Go | élevée | les deux échouent | ADR-006, un agent par régime |

---

## 3. Idées à tester (au-delà de l'architecture de base)

### I1 — Spéculation par échelle de précision

Utiliser une vue 1 bit des **mêmes** poids comme draft, et la vue 1,58 bit
comme vérificateur. Coût mémoire additionnel : ~0 (les poids sont déjà là,
la vue 1 bit est un sous-produit du packing). Gain attendu : ×2 à ×3 à
batch 1 (EAGLE-3 : 3,0-6,5×, arXiv 2503.01840 ; acceptation 0,82-0,88).

**Pourquoi c'est intéressant ici :** dans un régime calcul-borné, la
spéculation ne réduit pas le travail par token — elle le **multiplie**
(γ+1 passes par pas). Elle n'aide que si le draft est presque gratuit, ce qui
est le cas d'une vue 1 bit (¼ du coût). **À vérifier : le gain net peut être
négatif.** C'est précisément ce qui rend l'expérience intéressante.

### I2 — Routage dynamique par difficulté

Un petit classifieur choisit `top_k ∈ {1, 2, 4, 10}` selon la confiance du
routeur. Les tokens faciles coûtent 10× moins. À débit constant, on peut
allouer plus de capacité aux tokens difficiles.

**Risque :** le débit devient irrégulier, et la latence pire sur les tokens
difficiles. **Test :** distribution de `top_k` sur 10 000 tokens réels +
perte de qualité à budget moyen constant.

### I3 — Partager les projections d'attention linéaire entre couches

Les 48 couches DeltaNet coûtent 0,50 Go/token, soit 24 % du budget. Elles
sont redondantes par construction (même dimension, même rôle). Quatre
projections partagées par groupe de 4 couches ramèneraient le coût à
0,125 Go/token → **+18 % de débit**, au prix d'une perte de diversité.

**Test :** ablations 1 / 2 / 4 / 8 couches par groupe, 20 Md tokens.

### I4 — KV cache en mémoire externe (offload du contexte long)

Au-delà de 64 k, le KV coûte 0,3 Go/token. On peut le stocker en RAM et n'en
relire que les pages utiles (attention éparse, top-2 048 tokens). TurboQuant
(3-3,5 bits, arXiv 2504.19874) et NSA/MoBA ouvrent cette voie.

**Gain :** contexte 1 M tokens avec 0,1 Go/token. **Risque :** perte de rappel
précis. **Test :** RULER à 128 k et 1 M.

### I5 — Le routeur comme produit

80 % des requêtes n'ont pas besoin de 372 B. Un cascade 2 B → 215 B → 420 B
avec un classifieur à 1 M de paramètres ferait tomber le coût moyen à
~0,15 J/token au lieu de 2,00. C'est un facteur 13, sans toucher au modèle.

**C'est la seule idée de cette liste qui soit immédiatement rentable.** Elle
appartient à W4 et devrait être prioritaire sur toute optimisation
d'architecture.

### I6 — Distillation « backward » : le petit modèle enseigne le grand

Contre-intuitif : utiliser un 75 M spécialisé et très fiable sur un domaine
pour générer des traces, et distiller le 372 B dessus. Le grand modèle
apprend la **forme** du raisonnement court, le petit évite les erreurs du
grand.

**Statut :** spéculatif. Aucune référence solide. À ne pas financer avant
E-K2.

---

## 4. Questions ouvertes pour les autres agents

| # | Question | Destinataire | Pourquoi elle bloque |
|---|---|---|---|
| Q1 | La cible est-elle 100 Mo ou 100 Go ? | `9c2ae177`, humain | deux documents, deux budgets |
| Q2 | Quel packing ternaire réel (bits effectifs, échelles, buffers) ? | `Silex` | ±10 % sur le budget mémoire |
| Q3 | Quel corpus pour 50-100 Md tokens de réparation ? | W2 (à ouvrir) | sans données, pas de chemin A |
| Q4 | Quel index de retrieval tient dans 30 Go ? | W4 (à ouvrir) | c'est ce qui rend 215 B utile |
| Q5 | Adopter `GOPS ≥ 16·BW/b` comme test de conception ? | W5 | évite d'optimiser la mémoire quand le calcul bloque |
| Q6 | Mesurer `J/réponse juste` sur les deux régimes ? | W5, W6 | seule métrique comparable |

---

## 5. Ce qui me ferait abandonner HECATON

Écrit à l'avance, pour que personne ne puisse le découvrir après coup :

1. **E-K2 échoue** (ternaire > 5 points de perte à 1 B) → il faut 4 bits →
   la capacité tombe à 180 B et le débit à 25 tok/s → **pas d'avantage sur
   Qwen3-Next quantifié**. Abandon de l'architecture, conservation du
   calculateur et des mesures.
2. **E-K3 échoue** (noyau < 400 GOP/s) → le débit ne permet pas l'interaction
   → **HECATON devient un objet de laboratoire**, pas un produit.
3. **E-K8 échoue** (Qwen3-Next ternaire perd > 10 points) → le chemin A est
   mort → tout le régime 100 Go perd son intérêt pratique à court terme.
4. **E-K9 échoue** (le 75 M a un meilleur `J/réponse juste`) → HECATON n'est
   pas un produit ; c'est un programme de recherche sur la capacité.

Dans les quatre cas, **le travail garde de la valeur** : la loi de budget, le
calculateur et les deux microbenchmarks restent vrais et réutilisables.
C'est le test d'un bon programme de recherche : même faux, il apprend
quelque chose.
