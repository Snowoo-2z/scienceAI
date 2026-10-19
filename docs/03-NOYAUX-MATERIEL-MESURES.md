# Noyaux, matériel et mesures — ce qui limite vraiment

**KAIROS · 2026-10-19 · MESURÉ (VM 2 vCPU) + CALCULÉ + sources**

---

## 1. Pourquoi j'ai mesuré au lieu d'estimer

Toute la littérature sur les MoE locaux répète deux choses :

1. « le décodage est memory-bound » ;
2. « les experts sont dispersés, il faut les réordonner (expert clustering) ».

La première est vraie à 4-8 bits, fausse à 1,58 bit (voir `docs/01` §2).
La seconde n'avait, à ma connaissance, jamais été testée sur la granularité
réelle d'un expert (1-4 Mo). J'ai écrit deux microbenchmarks pour trancher.

**Avertissement de mesure.** L'environnement disponible est une VM à
**2 vCPU, 3 Go de RAM, sans swap**. Les valeurs absolues ne sont pas
représentatives d'une machine cible. Les **ratios** le sont : c'est eux qui
portent la conclusion. Le protocole est reproductible en une commande.

---

## 2. `bwprobe` — bande passante et localité

`gcc -O3 -march=native -pthread -o bwprobe tools/bwprobe.c && ./bwprobe 1024`

Tas de 1 Go, 2 threads, meilleur de 3 passes. Sortie brute :
`results/bwprobe.txt`.

### 2.1 Lectures de blocs aléatoires

| taille de bloc | Go/s | % du séquentiel |
|---:|---:|---:|
| 64 o | 3,38 | **13 %** |
| 256 o | 8,43 | 33 % |
| 1 Ko | 14,41 | 56 % |
| 4 Ko | 20,70 | 80 % |
| 16 Ko | 25,08 | 97 % |
| 64 Ko | 26,85 | **104 %** |
| 1 Mo | 26,91 | 104 % |
| 4 Mo | 27,66 | 107 % |

**Interprétation.** Le préfetcheur matériel et le parallélisme mémoire font
tout le travail dès que le bloc dépasse quelques kilo-octets. À 64 o, on paie
la ligne de cache et la latence DRAM : **7,6 fois plus lent**.

### 2.2 Gather de 10 experts (banque de 512 experts)

| taille d'expert | ordre du routeur | indices triés |
|---:|---:|---:|
| 256 Ko | 27,31 Go/s (106 %) | 27,17 Go/s (105 %) |
| 1 Mo | 27,47 Go/s (107 %) | 27,24 Go/s (106 %) |
| 4 Mo | 27,05 Go/s (105 %) | 27,50 Go/s (107 %) |

**Deux résultats négatifs, donc deux économies :**

1. **L'« expert clustering » ne sert à rien** à cette granularité. Dispersé ou
   contigu, l'écart est de ±2 %. Un expert de 4 Mo est lu en séquentiel par le
   contrôleur mémoire, quelle que soit son adresse. Les articles qui
   recommandent de réordonner les experts parlent de banques de 100 Mo à
   granularité de 64 o — pas de notre cas.
2. **Trier les indices d'experts n'aide pas** (ni ne nuit). C'est une
   optimisation gratuite en théorie, inutile en pratique sur ce matériel.

### 2.3 Où la localité compte vraiment

Le vrai danger est **en dessous de 4 Ko** : 20 à 65 % de bande passante perdue.
Cela concerne :

- le **KV cache paginé** (pages de 64 à 512 o) ;
- les **tables d'embedding** (un vecteur de 2 à 4 Ko par token) ;
- les **scales et zéros de quantification** (quelques octets par groupe) ;
- le **routeur** (un vecteur de sortie par couche).

**Recommandation :** imposer un alignement ≥ 4 Ko pour tout tenseur lu à chaque
token, et un packing de quantification par blocs de 256 poids minimum. Sinon,
le gain d'un noyau rapide est mangé par la mémoire.

---

## 3. `tgemv` — le débit réel d'un noyau ternaire

`gcc -O3 -march=native -o tgemv tools/tgemv.c && ./tgemv 2048 1024 100`

Matrice d'un expert HECATON : 2 048 × 1 024, poids 2 bits packés. Sortie brute :
`results/tgemv.txt`.

| implémentation | GOP/s | facteur |
|---|---:|---:|
| A. naïve (1 poids à la fois) | 5,35 | 1,0× |
| B. table de 256 entrées | 6,62 | 1,2× |
| C. déballage int8 + dot product (auto-vectorisé) | **8,49** | **1,6×** |

Sur 2 vCPU, un noyau C correct fait **8,5 GOP/s**. Extrapolé à 16 cœurs (si le
passage à l'échelle est linéaire, ce qui est optimiste) : ~68 GOP/s. Un noyau
AVX-512 écrit à la main, avec LUT vectorielle ou `vpdpbusd`, fait 4 à 10×
mieux : **300 à 700 GOP/s** sur un desktop 16 cœurs.

Le seuil pour être mémoire-borné à 1,7 bit et 166 Go/s est de **1 564 GOP/s**.
**On en est à 2 à 5 fois en dessous.** D'où la conclusion :

> **Le noyau ternaire est le chemin critique du projet, pas l'architecture.**
> Un mauvais noyau transforme un design à 80 tok/s en design à 20 tok/s, sans
> qu'aucune ligne du modèle ne change.

### 3.1 Ce qu'il faut construire (W3)

| Priorité | Travail | Gain attendu |
|---|---|---|
| 1 | GEMV ternaire AVX-512 (déballage + `vpdpbusd`), 16 cœurs | ×10 vs scalaire |
| 2 | LUT 3 bits par registre (style bitnet.cpp) | ×1,5-2 |
| 3 | Fusion déquant + matmul (pas de buffer intermédiaire) | ×1,2 |
| 4 | Préfetch explicite des experts top-k (10 flux) | ×1,1 |
| 5 | Attention MLA en 4 bits (noyau séparé) | débloque le prefill |

Le point 1 est le seul qui soit bloquant. Les points 2-5 sont du réglage.

---

## 4. Le matériel disponible en octobre 2026

| Machine | RAM | BW crête | BW utile (×0,65-0,72) | Prix | Verdict |
|---|---:|---:|---:|---:|---|
| PC DDR5-5600, 2 canaux | 128 Go | 96 Go/s | 60 | ~1 200 € | plancher ; calcul-borné |
| **Ryzen AI Max+ 395 (Strix Halo)** | **128 Go** | **256 Go/s** | **166** | **3 500 €** | **meilleur rapport** ; iGPU 50 TOPS pour le prefill |
| Mac Studio M4 Max | 128 Go | 546 Go/s | 393 | ~4 000 € | 1,5× plus rapide, même prix |
| Mac Studio M3 Ultra | 192 Go | 819 Go/s | 590 | 5 500 € | le plus rapide sous 6 k€ |
| Bi-EPYC, 12 canaux | 768 Go | 460 Go/s | 276 | 15 000 € | piège : RAM sans noyaux |
| RTX 5090 + 128 Go DDR5 | 128 Go | hybride | ~66 | 4 000 € | mauvais en énergie (14 J/token) |

**Recommandation :** Strix Halo 128 Go en premier (iGPU pour le prefill, CPU
pour les experts, 10 GbE pour servir). Mac Studio M4 Max si le budget le
permet : 1,5× le débit pour 15 % de prix en plus. **Ne pas acheter de
bi-Xeon** : beaucoup de RAM, peu de noyaux ternaires, 400 W.

> Les prix sont des relevés d'octobre 2026 : Strix Halo 128 Go entre 3 449 $
> et 3 879 $ selon le constructeur ; Mac Studio M4 Max 128 Go ~3 500 $ ;
> M3 Ultra 96 Go 5 299 $ (prix relevé après la hausse de juin 2026).

---

## 5. L'énergie : trois régimes, un seul levier

`J/token = octets × 0,08 J/Go + FLOPs × 0,1 pJ + P_plateforme / débit`

| Poste | HECATON-420A6, Strix Halo |
|---|---:|
| DRAM (2,05 Go) | 0,166 J |
| calcul (12,2 GFLOP) | 0,001 J |
| plateforme (90 W / 49 tok/s) | 1,834 J |
| **total** | **2,000 J/token** |
| part du modèle | **8,3 %** |
| réponse de 400 tokens | 800 J = 0,22 Wh |

**Trois conséquences, dont une qui contredit l'intuition :**

1. **L'énergie est à 92 % de la plateforme**, exactement comme dans le régime
   100 Mo (où la part du modèle est de 0,2 %). Les deux branches convergent :
   `E ≈ P/débit`. **Optimiser les joules, c'est optimiser le débit.**
2. **Le batching n'aide pas quand on est calcul-borné.** En batch 8, le débit
   par token ne change pas (mêmes MAC par token), donc l'énergie non plus.
   Il aide seulement quand on est mémoire-borné (Qwen3-Next : 1,59 → 1,51
   J/token).
3. **Le levier énergétique est le rapport GOPS/watt du noyau.** Passer de 800
   à 1 600 GOP/s double le débit à puissance constante : 2,00 → 1,08 J/token.
   C'est plus efficace que n'importe quelle optimisation du modèle.

### 5.1 Comparaison honnête avec la branche 100 Mo

| | NanoFR-75M | Qwen3-Next-80B q4.5 | HECATON-420A6 |
|---|---:|---:|---:|
| RAM | 0,2 Go | 46,7 Go | 84,9 Go |
| débit | 537 tok/s* | 84 tok/s | 49 tok/s |
| J/token | 0,18 | 1,23 | 2,00 |
| capacité | 73 M | 79 B | 372 B |

\* plafond roofline non atteignable (coûts fixes).

**Le petit modèle est 10× plus économe par token.** Mais il ne peut pas faire
la même chose. La bonne métrique est `J / réponse juste` : si le 372 B réussit
du premier coup ce que le 75 M rate trois fois sur quatre, le calcul s'inverse.
**Personne ne le sait aujourd'hui — c'est E-K9.**

---

## 6. Ce qu'il faut mesurer sur la machine cible (protocole)

| # | Mesure | Commande | Critère |
|---|---|---|---|
| M1 | BW séquentielle et par blocs | `./bwprobe 8192` | plateau ≥ 80 % dès 16 Ko |
| M2 | GOP/s ternaire 16 cœurs | `./tgemv 4096 4096 200` | ≥ 800 GOP/s |
| M3 | GOP/s du noyau de production | `llama-bench -m … -p 512 -n 128` | ≥ 600 GOP/s équivalent |
| M4 | RSS réel à 32 k | `/usr/bin/time -v` + `smem` | ≤ 88 Go |
| M5 | énergie au mur | prise connectée / RAPL | ≤ 2,2 J/token |
| M6 | prefill 1 k et 8 k | `llama-bench` | ≤ 2 s pour 1 k |

**Règle des six champs** (imposée par `AGENTS.md`) : modèle · quantification ·
contexte · backend · matériel · date. Sans les six, le chiffre n'est pas
comparable et ne doit pas entrer dans une décision.

---

## 7. Sources

- BitNet b1.58 et bitnet.cpp : arXiv 2410.16144 ; github.com/microsoft/BitNet
  (noyaux ternaires *lossless*, 2,37-6,17× sur x86, démo 100 B sur 1 CPU).
- KTransformers : SOSP'25 (671 B Q4 sur 1 GPU 24 Go + 382 Go DRAM, 13,6-14 tok/s).
- Roofline : Williams, Waterman & Patterson (2009) ; application au décodage
  (intensité arithmétique ≈ batch size).
- Prix et bande passante : relevés constructeurs, octobre 2026
  (`docs/REFERENCES.md`).
- Mesures locales : `results/bwprobe.txt`, `results/tgemv.txt`.
