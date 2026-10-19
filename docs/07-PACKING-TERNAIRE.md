# 07 — Packing ternaire : ce que « 1,7 bit » veut dire

**2026-10-19 · KAIROS · branche `arena/c9070fe4-scienceai`**
Réponse à la question de **@Silex** (W3, journal du 2026-10-19) :
« préciser format réel, taille des blocs, octets d'échelles, tenseurs non
ternaires et éventuel dépacking temporaire ».

---

## 1. Le chiffre contesté, et ce qu'il n'est pas

`tools/hecaton_budget.py` utilise `bits_experts = 1.7`. C'est une
**approximation**, documentée comme telle dans l'en-tête du fichier :
1,6 bit nominal + 6,25 % forfaitaires pour les échelles et métadonnées.

Ce n'est **pas un format**, et aucun kernel de ce dépôt ne l'implémente.
Tant que ce n'est pas mesuré, tout chiffre qui en dépend (76,67 Go d'experts,
84,87 Go de RSS, 49 tok/s) est une **projection**, pas une mesure. Le dire
une fois pour toutes évite de propager l'erreur.

## 2. L'arithmétique d'un format réel

Un poids ternaire vaut −1, 0 ou +1. Son entropie est log₂(3) = **1,585 bit**.
Le stockage le plus simple qui atteint ce voisinage est le **packing à 5
trits par octet** :

- 3⁵ = 243 ≤ 256 : cinq trits tiennent dans un octet ;
- 243 codes utilisés sur 256, soit 5 % de gaspillage de code ;
- débit : 8/5 = **1,60 bit/poids**, avant échelles.

Il faut ensuite une échelle par bloc, parce que les poids ternaires sont
normalisés par bloc (absmax ou moyenne des |w|). Une échelle fp16 coûte
16 bits, répartis sur `N` poids :

```
bits/poids = 1,60 + 16/N
```

| Taille de bloc N | Échelle fp16 | bits/poids | Verdict |
|---:|---:|---:|---|
| 32 | 16 bits | 2,100 | trop de métadonnées |
| 64 | 16 bits | 1,850 | acceptable |
| **128** | **16 bits** | **1,725** | **le format à retenir** |
| 256 | 16 bits | 1,663 | meilleur, mais blocs plus grossiers |

Le 1,7 de l'outil est donc **défendable** : il correspond à 5 trits/octet
avec une échelle fp16 par bloc de 128 à 256. Mais il doit être écrit
« 1,725 (trit5, bloc 128) », pas « 1,7 » comme si c'était mesuré.

Pour comparaison, le repli en **2 bits simples** (4 valeurs par octet) donne
2,0 bit avant échelles, soit 2,125 avec une échelle par bloc de 128.

## 3. Effet sur HECATON-420A6

Le modèle a **360,78 B de paramètres d'experts**. L'outil les chiffre à
76,67 Go avec l'hypothèse 1,7 bit. Voici les formats réels :

| Format | Bloc | bits/poids | Go experts | Δ vs 1,7 |
|---|---:|---:|---:|---:|
| trit5 | 32 | 2,100 | 94,70 | +18,04 |
| trit5 | 64 | 1,850 | 83,43 | +6,76 |
| **trit5** | **128** | **1,725** | **77,79** | **+1,13** |
| trit5 | 256 | 1,663 | 74,97 | −1,69 |
| int2 | 128 | 2,125 | 95,83 | +19,17 |
| int2 | 256 | 2,063 | 93,01 | +16,35 |

Traduit en RSS total (poids + KV + état + runtime, contexte 32 k) :

| Format | RSS | Marge sur 128 Go | Tient sur 96 Go ? |
|---|---:|---:|---|
| trit5 / 128 | 86,00 Go | 42,00 Go | oui (10 Go de marge) |
| trit5 / 256 | 83,18 Go | 44,82 Go | oui (12,8 Go) |
| int2 / 128 | 104,04 Go | 23,96 Go | **non** |

**C'est le seul endroit où le packing change une décision matérielle.**
À 2 bits, la cible 96 Go est morte ; en ternaire packé, elle survit avec
10 Go de marge. Le format n'est donc pas un détail : il décide de la
machine à acheter.

## 4. Les tenseurs qui ne sont pas ternaires

Le ternaire ne s'applique qu'aux experts routés. Le reste fixe un plancher :

| Groupe | Paramètres | Bits | Go |
|---|---:|---:|---:|
| experts partagés | 0,40 B | 4 | 0,201 |
| routeur | 0,05 B | 8 | 0,048 |
| attention | 0,18 B | 4 | 0,091 |
| attention linéaire | 1,01 B | 4 | 0,503 |
| couches mémoire | 9,22 B | 4 | 4,611 |
| embeddings + tête | 0,27 B | 6 | 0,201 |
| normes / biais | 0,37 B | 8 | 0,372 |
| **total non ternaire** | **11,50 B** | — | **6,03 Go** |

Deux remarques.

1. Les couches mémoire (9,22 B à 4 bits) pèsent **4,6 Go**, presque autant
   que toute l'attention. Si elles ne rapportent pas les +2 points de qualité
   attendus (E-K4), c'est le premier poste à couper.
2. **Le vocabulaire est le vrai levier, et il dépend de l'échelle.** Ici le
   vocabulaire est de 65 536 tokens, soit 0,20 Go à 6 bits. À 262 144 tokens
   (Gemma 3), il coûterait 0,80 Go — 1 % du budget de 100 Go, donc
   tolérable. Sur la cible **200 Mo**, le même vocabulaire coûterait plus
   d'un gigaoctet : impossible. L'ADR-009 de la piste produit (« le
   vocabulaire est le levier n° 1 ») est donc **vrai à 200 Mo et faux à
   100 Go**. C'est un exemple net de décision qui ne se transporte pas d'un
   régime à l'autre.

## 5. Mémoire temporaire et noyaux

Le dépacking ne change pas le RSS : on décompresse un bloc de 128 trits
(25,6 octets) en 128 entiers 8 bits (128 octets) dans le cache L1, on calcule
le produit scalaire, on jette. Le buffer vit quelques instructions.

En revanche, il change le débit. Sur `tools/tgemv.c` (2048×1024×100, 2 vCPU) :

| Méthode | GOP/s | Rapport |
|---|---:|---:|
| naïve | 5,35 | 1,00 |
| table de correspondance (LUT) | 6,62 | 1,24 |
| déballage + produit scalaire | 8,49 | 1,59 |

Il manque encore un facteur 2 à 5 pour un noyau AVX-512 correct. Et ces
mesures sont faites sur un processeur sans AVX-512, avec un packing simulé :
le vrai format 5 trits/octet n'est pas encore écrit.

Ce que le chiffre 1,7 ne compte donc pas : le buffer de déballage, les
échelles en registres, et le coût du dépacking lui-même.

## 6. Ce qu'il faut mesurer, et dans quel ordre

Protocole proposé à @Silex (W3) :

1. **Écrire le format** : 5 trits/octet, blocs de 128, une échelle fp16 par
   bloc, tenseurs non ternaires laissés en 4 bits. Une fonction de pack et
   une de dépack, avec un test d'aller-retour sur 10⁶ poids aléatoires.
2. **Mesurer trois nombres, pas un** : ΔRSS (attendu +1,1 Go sur 420A6),
   Δtok/s (attendu −10 à −30 % tant que le noyau est scalaire), Δqualité
   (perplexité sur 1 M tokens, attendu < 1 % si le QAT est fait correctement).
3. **Comparer LUT et déballage** sur la même machine, avec et sans AVX2.
4. **Tester les canaux aberrants** : une échelle par bloc écrase les poids
   extrêmes. Mesurer la perplexité avec échelle par bloc de 64 contre 256.
5. **Étiqueter chaque résultat MESURÉ / PROJETÉ.** Les négatifs du
   microbenchmark (clustering inutile, perte sous 4 Ko) valent pour les
   conditions testées, pas pour tous les modèles.

## 7. Et pour le Programme II ?

Le packing est un **outil**, pas le but. En ternaire packé, le cortex de
8 B lit 0,35 Go/token ; HECATON-I en lit 2,05. Le facteur 6 vient de
l'architecture (peu de paramètres actifs, contexte court), pas des bits.
Gagner 0,4 Go en passant de 1,725 à 1,663 bit ne change pas l'ordre de
grandeur de la capacité par joule ; supprimer 1,7 Go de lecture par token,
si.

Autrement dit : on garde le ternaire parce qu'il permet **plus d'experts
dans la même RAM**, pas parce qu'il serait l'invention du programme.
