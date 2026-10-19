# HECATON — architecture d'un 372 B qui tient dans 85 Go

**KAIROS · 2026-10-19 · CALCULÉ + HYPOTHÈSES explicites**
Config : `configs/hecaton-420a6.json` · sortie : `results/hecaton-420a6.txt`

---

## 1. Objectif et non-objectif

**Objectif :** maximiser la capacité (paramètres stockés) et la qualité, sous
trois contraintes simultanées :

- ≤ 96 Go de RAM au pic, contexte 32 k ;
- ≥ 40 tok/s en décodage batch 1 sur une machine à moins de 4 000 € ;
- ≤ 2 J/token au mur, sans GPU.

**Non-objectif :** battre GPT-5. La cible honnête est la classe
gpt-oss-120b / Qwen3-235B-A22B, avec 4 à 6 fois moins d'octets lus par token.

## 2. Les six décisions, et pourquoi

### D1 — Sparsité extrême : 896 experts, top-10 (ρ = 1,6 %)

C'est la seule variable qui achète de la capacité **sans** coûter de bande
passante. À 96 Go, passer de 512 à 896 experts ajoute 75 % de paramètres pour
2 % de débit en moins.

Comparaison mesurée par le calculateur, même plateforme, même lecture :

| Variante | total | actifs | RAM | Go/token | tok/s (800 GOP/s) |
|---|---:|---:|---:|---:|---:|
| HECATON-215A6 | 213 B | 6,0 B | 49,5 Go | 2,05 | 49,4 |
| HECATON-420A6 | **372 B** | 6,0 B | **84,9 Go** | 2,05 | **49,1** |

**+75 % de capacité, −0,6 % de débit, +35 Go de RAM.** C'est le cœur du projet.

### D2 — Experts ternaires à 1,58 bit (1,7 bit effectif)

À 4 bits, 372 B de paramètres pèseraient 186 Go : impossible. À 1,7 bit,
77 Go. La question n'est pas « est-ce que le ternaire est bon ? » mais
« est-ce que la perte de qualité est inférieure au gain de capacité ×2,4 ? ».
C'est l'expérience E-K2, et elle n'a pas de réponse publique au-delà de 3 B.

### D3 — Attention hybride : 3 couches DeltaNet pour 1 couche MLA

Le KV cache est relu à chaque token. Avec 64 couches d'attention pleine en
GQA, le KV coûterait 0,6 Go/token à 32 k — 30 % du budget de lecture, pour
un gain de qualité incertain.

- 48 couches linéaires (Gated DeltaNet) : état constant, ~0 octet/token.
- 16 couches MLA (latent 512 + RoPE 64, 4 bits) : 0,15 Go/token au total.

Résultat : **0,15 Go de KV à 32 k**, au lieu de 0,6 à 1,6 Go pour du GQA
classique. Coût : 0,59 Go de paramètres d'attention, lus à chaque token.

### D4 — Couches mémoire plutôt que des experts supplémentaires

Six à huit couches de mémoire produit-clé (3 M de slots × 384 dimensions,
top-32) ajoutent **9,2 B de paramètres pour 4,6 Go de RAM et ~50 Ko de lecture
par token**. Le rapport capacité/bande passante est 10 000 fois meilleur que
celui d'un expert.

C'est la brique la plus sous-exploitée de la littérature : Meta rapporte
jusqu'à 128 B de paramètres mémoire, et un 1,3 B + mémoire égalerait
Llama2-7B à compute égal (arXiv 2412.09764). **HYPOTHÈSE :** à cette échelle,
elle reste stable et n'écrase pas le raisonnement. À tester (E-K4).

### D5 — Vocabulaire 131 k et routeur de rang 256

La tête de sortie coûte 0,20 Go/token (10 %). Le routeur dense coûterait
0,25 Go/token (12 %). Un routeur `d → 256 → E` ramène cela à 0,05 Go.
Deux optimisations invisibles dans les papiers, décisives ici.

### D6 — KV en 4 bits, sans exception

0,15 Go/token à 32 k. En 8 bits, 0,30 Go : +7 % de débit perdu pour un gain
de qualité non mesuré. TurboQuant (3,5 bits) et NVFP4 descendent encore plus
bas ; à réserver aux contextes > 64 k.

## 3. Budget mémoire complet (MESURÉ par le calculateur)

```
experts routés .............. 76,67 Go   (896 × 64 × 3 × 2048 × 1024 à 1,7 bit)
expert partagé ............... 0,20 Go
attention (MLA + DeltaNet) ... 0,59 Go
couches mémoire .............. 4,61 Go
embeddings ................... 0,20 Go
routeur ...................... 0,05 Go
normes / biais ............... 0,37 Go
                              ─────────
poids totaux ................ 82,69 Go   (1,78 bit/poids effectif moyen)
KV cache (32 k, MLA q4) ...... 0,15 Go
état récurrent ............... 0,03 Go
runtime ...................... 1,50 Go   ← HYPOTHÈSE, à mesurer (E-K1)
buffers / scratch ............ 0,50 Go   ← HYPOTHÈSE, à mesurer
                              ─────────
TOTAL ....................... 84,87 Go
marge sur 96 Go ............. 11,13 Go   (13 %)
```

La marge de 11 Go absorbe l'incertitude sur le runtime (facteur 3) et permet
un contexte de 64 k si nécessaire.

## 4. Ce qui est lu à chaque token

| Poste | Go/token | Part | Commentaire |
|---|---:|---:|---|
| experts actifs (10/896) | 0,84 | 41 % | incompressible : c'est le calcul utile |
| attention linéaire (48 couches) | 0,50 | 24 % | **le poste le plus sous-estimé** |
| expert partagé | 0,20 | 10 % | stabilité du routage |
| tête de sortie | 0,20 | 10 % | 131 k × 2 048 à 6 bits |
| KV relu | 0,15 | 7 % | MLA 4 bits |
| attention pleine (16 couches) | 0,09 | 4 % | MLA |
| routeur | 0,05 | 2 % | rang 256 |
| couches mémoire | ~0,00005 | ~0 % | 32 slots × 384 |
| **TOTAL** | **2,05 Go** | 100 % | |

**Enseignement inattendu :** les 48 couches linéaires coûtent **plus cher en
bande passante** que les experts actifs. Une architecture « efficace » façon
Mamba/DeltaNet réduit le KV mais ajoute des projections denses relues à chaque
token. À 100 Mo, c'est indifférent (le runtime domine) ; à 100 Go, c'est un
quart du débit. **Piste :** partager ces projections entre couches, ou
descendre à 1,7 bit dessus aussi (gain potentiel : 0,35 Go/token, +20 % de
débit).

## 5. Débit

Hypothèse de noyau : 800 GOP/s utiles (16 cœurs AVX-512, noyau ternaire
correct ; voir `docs/03`). `tools/hecaton_budget.py sensitivity` :

| plateforme | BW utile | GOPS | tok/s | goulot | J/token |
|---|---:|---:|---:|---|---:|
| PC DDR5 2 canaux | 60 Go/s | 300 | 18,4 | calcul | 3,15 |
| **Strix Halo 128 Go** | 166 Go/s | 800 | **49,1** | calcul | **2,00** |
| Mac Studio M4 Max | 393 Go/s | 1 200 | 73,6 | calcul | 1,05 |
| Mac Studio M3 Ultra | 590 Go/s | 1 800 | 110,4 | calcul | 0,98 |
| Bi-EPYC 12 canaux | 276 Go/s | 1 200 | 73,6 | calcul | 5,60 |
| Hybride RTX 5090 + DDR5 | 66 Go/s | 3 000 | 32,2 | mémoire | 14,2 |

**À 800 GOP/s, toutes les machines sont calcul-bornées.** Le débit est
`GOPS / 12,2 GFLOP`, indépendant de la mémoire. C'est le renversement central
de ce travail : **on n'achète pas de la vitesse avec de la bande passante, on
en achète avec des noyaux.**

Si le noyau atteint 1 600 GOP/s (AVX-512 VNNI bien écrit sur 16 cœurs), Strix
Halo passe à 81 tok/s et la mémoire redevient limitante.

## 6. Latence : le talon d'Achille

| | valeur |
|---|---:|
| décodage | 20,4 ms/token (49 tok/s) |
| **prefill 1 000 tokens** | **15,0 s** |
| temps avant le premier token | ~15 s |

Le prefill est **calcul-borné et parallélisable**, contrairement au décodage.
Sur un CPU à 800 GOP/s, 1 000 tokens de prompt coûtent 15 s. C'est rédhibitoire
pour un usage interactif.

Trois remèdes, par ordre de préférence :

1. **Faire le prefill sur l'iGPU/NPU.** Strix Halo embarque 50 TOPS INT8 :
   1 000 tokens passent de 15 s à ~0,3 s. Le décodage reste sur CPU.
2. **Raccourcir les prompts** (RAG, résumé, cache de préfixe). Un prompt de
   200 tokens coûte 3 s.
3. **Chunked prefill** avec recouvrement : ne change pas le total, améliore
   la latence perçue.

C'est un argument fort pour l'architecture **hybride** (GPU pour l'attention
et le prefill, CPU pour les experts) — exactement ce que fait KTransformers.

## 7. Variantes

| Variante | Usage | total | RAM | tok/s | Fichier |
|---|---|---:|---:|---:|---|
| HECATON-95A3 | station 32 Go, latence | 91 B | 22,3 Go | 92 | `hecaton-95a3.json` |
| HECATON-215A6 | machine 64 Go | 213 B | 49,5 Go | 49 | `hecaton-215a6.json` |
| **HECATON-420A6** | **machine 96-128 Go** | **372 B** | **84,9 Go** | **49** | `hecaton-420a6.json` |

Les trois partagent la même géométrie de lecture (même `d_model`, même top-k) :
seul le nombre d'experts change. C'est volontaire — cela permet de tester la
qualité à capacité croissante sans changer le noyau ni le débit.

## 8. Alternatives rejetées

| Alternative | Pourquoi rejetée |
|---|---|
| Dense 70 B à 4 bits | 35 Go, 45 tok/s... mais ρ=1, donc 0,35 Go/token et **aucune marge de capacité**. Un dense ne peut pas utiliser 96 Go sans devenir lent. |
| MoE à 64 gros experts top-8 | ρ = 12,5 %. À budget égal, 8× moins de capacité et 6× moins de débit. |
| Mamba/RWKV pur | Supprime le KV mais garde des projections denses relues à chaque token. Ne résout pas le problème de bande passante (cf. §4). |
| Streaming depuis le SSD (llm-in-a-flash) | 5 Go/s de NVMe = 0,4 tok/s pour 2 Go/token. Utile seulement si le modèle ne tient **pas** en RAM ; ici il tient. |
| Batching pour amortir les poids | Excellent pour le débit agrégé, mais ne réduit pas la latence et **n'aide pas** si l'on est calcul-borné. Un serveur, pas un assistant. |
| 1 bit (binaire) | Divise encore la mémoire par 1,6, mais la littérature ne montre pas de qualité acceptable sans distillation lourde. À surveiller (BitNet a1.58, pas a1.0). |

## 9. Ce qui falsifierait ce design

1. **E-K2 :** si un 700 M ternaire perd > 5 points vs son équivalent 4 bits à
   budget égal, l'hypothèse D2 s'effondre → il faut 4 bits → capacité max
   ~180 B à 96 Go.
2. **E-K3 :** si le noyau ternaire réel plafonne à 300 GOP/s sur 16 cœurs, le
   débit tombe à 25 tok/s → la cible « ≥ 40 tok/s » échoue.
3. **E-K4 :** si les couches mémoire dégradent le raisonnement (au lieu
   d'ajouter de la connaissance factuelle), il faut les retirer → −9 B de
   capacité et −4,6 Go.
4. **E-K6 :** si l'attention hybride 1/4 coûte > 3 points sur le rappel long
   contexte, il faut remonter à 1/2 → KV ×2 et −7 % de débit.
5. **E-K7 :** si le prefill CPU ne descend pas sous 2 s pour 1 000 tokens, le
   produit n'est pas interactif sans GPU.

**Si 1 et 3 échouent ensemble, HECATON n'a plus d'avantage sur Qwen3-Next-80B
quantifié — et il faut l'abandonner.** C'est écrit ici pour que personne ne
puisse le découvrir après coup.
