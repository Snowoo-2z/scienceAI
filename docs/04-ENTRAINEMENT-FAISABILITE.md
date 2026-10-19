# Entraînement et faisabilité — le mur n'est pas là où on l'attend

**KAIROS · 2026-10-19 · CALCULÉ, avec sources pour les comparaisons**

---

## 1. L'arithmétique du calcul

`FLOPs ≈ 6 · N_actif · D` (Chinchilla, arXiv 2203.15556, adapté aux MoE où
seuls les paramètres actifs sont traversés).

Pour HECATON-420A6 (6,0 B actifs), à 396 TFLOPS utiles
(H100 990 TFLOPS × MFU 0,40) et 2,50 $/h :

| tokens | FLOPs | heures H100 | coût | tokens / param actif |
|---:|---:|---:|---:|---:|
| 120 Md | 4,3 × 10²¹ | 3 000 | 7 600 $ | 20 (Chinchilla) |
| 500 Md | 1,8 × 10²² | 12 600 | 31 500 $ | 83 |
| 1 T | 3,6 × 10²² | 25 200 | 63 000 $ | 167 |
| 2 T | 7,2 × 10²² | 50 400 | 126 000 $ | 333 |
| 10 T | 3,6 × 10²³ | 252 000 | 630 000 $ | 1 667 |

**Première conclusion :** entraîner ce modèle n'est pas délirant pour un
laboratoire, mais c'est **deux à trois ordres de grandeur** au-dessus du
budget de ce projet. Ce n'est pas l'inférence qui coûte cher, c'est le
pré-entraînement.

## 2. Le vrai mur : les tokens par expert

Le calcul ci-dessus ignore un fait que les papiers MoE passent sous silence :
**chaque expert ne voit qu'une fraction des tokens**, et un expert sous-entraîné
est un expert inutile.

| Modèle | couches MoE | experts | experts (couches × experts) | tokens | **tokens / expert** |
|---|---:|---:|---:|---:|---:|
| DeepSeek-V3 | 58 | 256 | 14 848 | 14,8 T | **997 M** |
| Qwen3-Next-80B | 48 | 512 | 24 576 | 15 T | **610 M** |
| Mixtral 8×7B | 32 | 8 | 256 | ~1 T | 3 900 M |
| **HECATON-215A6** | 64 | 512 | 32 768 | 2 T | **61 M** |
| **HECATON-420A6** | 64 | 896 | 57 344 | 2 T | **35 M** |

**HECATON-420A6 aurait 17 à 28 fois moins de tokens par expert que les modèles
de référence.** À 35 M tokens par expert, un expert de 6,3 M paramètres n'a
pas les moyens d'apprendre quoi que ce soit de spécialisé : il reste une
copie bruitée de son voisin.

**Deuxième conclusion, sévère :** la capacité de 372 B est **structurellement
inatteignable** avec 2 T tokens. Il faudrait 10 à 20 T tokens, soit
600 k$ à 1,2 M$ — hors de portée.

Ce n'est pas un défaut de l'architecture : c'est une loi de la sparsité. **On
ne peut pas augmenter indéfiniment le nombre d'experts sans augmenter les
données proportionnellement.** Les modèles à 1 000 experts qui existent (et ils
existent) ont été entraînés sur des corpus de 20 T+ tokens.

## 3. Trois chemins, trois verdicts

### Chemin A — Quantifier et réparer un modèle existant (recommandé, immédiat)

1. Prendre **Qwen3-Next-80B-A3B** (80 B total, 3 B actifs, 512 experts,
   15 T tokens d'entraînement, qualité connue).
2. Le quantifier en ternaire (1,58 bit) sur les experts, 4 bits sur le reste.
3. Réparer par QAT sur 50 à 100 Md tokens de distillation.
4. Mesurer la perte réelle (E-K2).

| étape | compute | coût |
|---|---:|---:|
| ternarisation + QAT 100 Md tokens | 1 500 H100-h | 3 800 $ |
| distillation (traces déjà générées par l'enseignant) | inclus | — |
| **total** | | **≤ 5 000 $** |

**Résultat attendu :** 23 Go de poids, ~84 tok/s, qualité −2 à −8 points
(inconnu). **C'est le seul chemin que ce projet peut financer**, et il donne
un objet réel à mesurer en quelques semaines.

### Chemin B — Upcycling + ternarisation (intermédiaire)

Partir d'un MoE existant et dupliquer les experts pour remplir la mémoire
(sparse upcycling, arXiv 2212.05055 ; Llama3-8B → 8 experts top-2 pour < 1 %
du compute de pré-entraînement, +2 pts MMLU, arXiv 2412.09952).

| étape | compute | coût |
|---|---:|---:|
| dupliquer 512 → 896 experts, 50 Md tokens | 750 H100-h | 1 900 $ |
| QAT + réparation, 150 Md tokens | 2 250 H100-h | 5 600 $ |
| **total** | | **7 500 $** |

**Résultat attendu :** 372 B nominal, mais les 384 experts dupliqués sont des
copies. Le gain de capacité est **cosmétique** jusqu'à ce qu'ils se
différencient, ce qui demande beaucoup plus de tokens. **Verdict : ne pas
faire avant d'avoir mesuré le chemin A.**

### Chemin C — Pré-entraînement from scratch (programme de recherche)

2 à 10 T tokens, 126 k$ à 630 k$, 6 à 18 mois. C'est le seul chemin qui
produit réellement 372 B de capacité utile. **Verdict : hors périmètre pour
ce projet, à documenter pour un financement futur.**

### Chemin D — Entraînement décentralisé (pari)

DiLoCo permet 8 travailleurs à qualité synchrone avec **500× moins de
communication** (arXiv 2311.08105) ; Streaming DiLoCo descend à 4 bits et 400×
moins d'octets (arXiv 2501.18512) ; DiLoCoX pré-entraîne un 107 B sur un lien
**1 Gbit/s** (arXiv 2506.21263).

Un réseau de 50 machines à 128 Go (Strix Halo, Mac Studio) représente
6,4 To de RAM et ~8 To/s de bande passante agrégée. En théorie de quoi
pré-entraîner un 300B-A6B sur 2 T tokens en un an. En pratique :

- la bande passante agrégée n'est **pas** le facteur limitant : c'est la
  latence et la fiabilité des nœuds ;
- 50 bénévoles qui débranchent leur machine coûtent plus cher que 50 GPU
  loués pendant trois mois ;
- la gouvernance des données devient le problème principal.

**Verdict :** crédible pour un consortium, pas pour un projet solo. À garder
comme option si 50+ volontaires se manifestent.

## 4. Le verdict, en une phrase

> **On peut faire tourner un 372 B dans 85 Go. On ne peut pas l'entraîner
> ici. Donc la première étape n'est pas de construire HECATON, c'est de
> quantifier Qwen3-Next-80B en ternaire et de mesurer ce que ça coûte en
> qualité.**

C'est moins spectaculaire que « un LLM frontière dans 100 Go », mais c'est
vrai, et c'est mesurable en quelques semaines avec 5 000 $.

## 5. La recette QAT, si l'on veut tenter le ternaire

1. **Rotations** (Hadamard / QuaRot) avant ternarisation, pour casser les
   valeurs aberrantes. Sans rotation, le ternaire échoue.
2. **Échelles par groupe de 128 à 256 poids**, en FP16, avec décalage
   (absmean scaling).
3. **Straight-through estimator** sur les poids ternaires.
4. **Distillation** de l'enseignant (KL sur les logits + cross-entropy sur les
   tokens), pas seulement du fine-tuning supervisé.
5. **Apprentissage des activations en 8 bits** : le ternaire ne concerne que
   les poids.
6. **Réparation ciblée** : sur les couches d'attention d'abord (elles coûtent
   cher en bande passante), les experts ensuite.

Pièges connus :

- les embeddings et la tête de sortie **ne doivent pas** être ternarisés ;
- les experts dupliqués par upcycling doivent être **bruités** avant
  réparation, sinon ils restent identiques et n'apportent rien ;
- la précision effective du fichier (avec échelles) est 1,7 bit, pas 1,58 :
  compter +7 % de RAM.

## 6. Données et distillation (question ouverte pour W2)

Pour le chemin A, il faut 50 à 100 Md tokens de traces de qualité. Trois
sources possibles :

| source | volume | coût | qualité |
|---|---:|---:|---|
| Génération par un enseignant frontière (API) | 100 Md | 50 k$ à 500 k$ | la meilleure |
| Corpus ouvert (FineWeb-Edu, DCLM, Pile) + SFT | illimité | ~0 | bonne pour le pré-entraînement, faible pour le raisonnement |
| Distillation depuis Qwen3-Next lui-même (auto-distillation) | 100 Md | ~0 | ne peut pas dépasser l'enseignant |

**Recommandation :** commencer par l'auto-distillation (gratuite) pour valider
la chaîne, puis acheter 5 à 10 Md tokens de traces ciblées sur le
raisonnement (≈2 500 à 25 000 $). C'est le meilleur rapport qualité/euro du
projet, conformément à ce que dit déjà le rapport de la branche 100 Mo.

## 7. Ce que ça change pour le matériel

Si la cible réaliste est **215 B / 49,5 Go**, une machine à 64 Go suffit
(Strix Halo 64 Go à ~2 200 €, ou un PC DDR5 128 Go à 1 200 €). Les 45 Go
restants sur une machine 128 Go servent à :

- un contexte de 128 k (KV MLA : 0,6 Go) ;
- un index de retrieval de 20 à 30 Go (10 à 20 M de passages) ;
- un cache de poids chauds et le système d'exploitation.

**C'est une meilleure allocation que 85 Go de poids mal entraînés.** Elle
correspond exactement à ce que recommande la branche 100 Mo, à une autre
échelle : la connaissance va dans l'index, pas dans les poids.
