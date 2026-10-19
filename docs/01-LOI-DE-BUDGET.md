# La loi de budget — une seule arithmétique, de 100 Mo à 200 Go

**KAIROS · 2026-10-19 · CALCULÉ (reproductible via `tools/hecaton_budget.py`)**

---

## 1. Les cinq équations

Tout le reste en découle. `P` = paramètres, `b` = bits effectifs,
`T` = contexte, `ρ` = fraction des poids lue par token.

```
(1) RAM      = Σ_groupe P_g · b_g / 8   +   KV(T)   +   état   +   runtime   +   buffers
(2) KV(T)    = n_couches_attention · T · n_valeurs_par_couche · b_kv / 8
(3) octets/t = P_actif · b / 8  +  KV_relu(T)  +  tête_de_sortie
(4) tok/s    = min( BW_utile / octets_par_token ,  GOPS_utiles / FLOPs_par_token )
(5) J/token  = octets · e_DRAM  +  FLOPs · e_op  +  P_plateforme / débit
```

et les deux quantités dérivées qui servent à concevoir :

```
ρ = P_actif / P_total                 (taux d'activation)
Λ = RAM / octets_par_token            (levier de capacité)
```

`Λ` a une lecture directe : **combien de tokens la mémoire peut contenir**. À
RAM et bande passante fixées, le débit est proportionnel à `Λ` :

```
tok/s = BW_utile · Λ / RAM
```

## 2. Le résultat central : la mémoire ne limite plus, le noyau limite

En décodage batch=1, l'intuition dominante est « memory-bound ». Elle est
**fausse dès qu'on descend sous 2 bits**. La comparaison des deux temps :

```
temps mémoire  = (P · b / 8) / BW
temps calcul   = (2 · P) / GOPS
```

Le calcul l'emporte quand `2·P/GOPS > P·b/(8·BW)`, c'est-à-dire :

```
        GOPS  <  16 · BW / b            ← LE SEUIL À RETENIR
```

| bits | BW utile | GOPS nécessaires pour être mémoire-borné |
|---:|---:|---:|
| 1,7 | 96 Go/s (DDR5 2 canaux) | **904** |
| 1,7 | 166 Go/s (Strix Halo) | **1 564** |
| 1,7 | 393 Go/s (M4 Max) | **3 700** |
| 4,0 | 166 Go/s | 664 |
| 4,5 | 166 Go/s | 590 |

**Conséquence :** à 1,58 bit, un noyau scalaire en C (8,5 GOP/s sur 2 vCPU,
mesuré) est **180 fois trop lent**. Même un excellent noyau AVX-512 à
800 GOP/s reste calcul-borné sur Strix Halo. La quantification extrême déplace
le problème de la bande passante vers le **débit de déquantification**.

C'est une bonne nouvelle déguisée : le calcul s'améliore avec les générations
de CPU, la bande passante beaucoup moins vite.

## 3. Ce que la loi donne, budget par budget

`tools/hecaton_budget.py law --bits 1.7` (extrait de `results/loi-budget.txt`,
contexte 8 192, MoE top-10, MLA) :

| budget | bits | total | actifs | Go/token | tok/s | Λ | experts | géométrie |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 Go | 1,7 | 27,4 B | 0,55 B | 0,20 | 814 | 39 | 768 | 40×768 |
| 32 Go | 1,7 | 131,7 B | 2,8 B | 0,96 | 174 | 34 | 768 | 12×3072 |
| 96 Go | 1,7 | 436,6 B | 4,3 B | 1,33 | 125 | 72 | 1536 | 80×1536 |
| 96 Go | 4,0 | 182,6 B | 3,6 B | 2,11 | 79 | 46 | 768 | 24×2560 |
| 192 Go | 1,7 | 875,5 B | 16,3 B | 4,92 | 34 | 39 | 768 | 80×3072 |

Trois lectures :

1. **Sous 1 Go, la roofline ne veut plus rien dire.** Les lignes à 0,5-1 Go
   sortent 2 000+ tok/s : c'est un artefact. Le tokenizer, l'échantillonnage et
   la latence de lancement coûtent plus que le modèle. C'est précisément le
   régime de l'autre branche, et c'est pourquoi son document parle de runtime
   et de vocabulaire, pas de bande passante.
2. **Les bits achètent de la capacité *et* de la vitesse**, parce qu'ils
   permettent d'acheter plus d'experts plutôt que plus de largeur. À 96 Go,
   1,7 bit donne 437 B de capacité pour 125 tok/s ; 4 bits donne 183 B pour
   79 tok/s. Le ternaire gagne sur les deux tableaux — **si** la qualité tient.
3. **Λ plafonne autour de 40-70.** On ne peut pas dépasser ~100 sans descendre
   sous 1 bit ou sans conteneur mémoire externe. C'est une borne dure.

> Attention : ce tableau maximise le **nombre de paramètres**, pas la
> **qualité**. Une géométrie 12×3072 avec 768 experts est un artefact
> d'optimisation, pas une recommandation. Les configurations défendues sont
> dans `docs/02-ARCHITECTURE-HECATON.md`.

## 4. Le pont avec la branche 100 Mo

La même équation (1) explique les deux régimes. Ce qui change est le terme
dominant.

| | 100 Mo | 1 Go | 96 Go |
|---|---:|---:|---:|
| Budget | 0,1 Go | 1 Go | 96 Go |
| Runtime incompressible | ~0,026 Go | ~0,15 Go | ~1,5 Go |
| **Taxe du runtime** | **26 %** | 15 % | **1,6 %** |
| KV à 2 k / 32 k | 0,01 / 0,17 Go | 0,03 / 0,48 Go | 0,15 Go (q4) |
| ρ typique | 1,0 (dense) | 1,0 (dense) | 0,016 (MoE) |
| Λ | ~1,9 | ~2 | **41** |
| Terme dominant | coûts fixes | compromis | noyau / bande passante |
| Levier principal | données + distillation | architecture | sparsité + quantification |

**La bascule n'est pas la taille, c'est le rapport `runtime / budget`.**
À 100 Mo, un runtime de 26 Mo mange un quart du budget avant tout poids utile :
il faut donc un modèle dense minuscule, spécialisé, et déporter la connaissance
dans un index. À 96 Go, ce même runtime vaut 1,6 % : on peut se permettre un
modèle 4 000 fois plus gros, très sparse, dont on ne lit que 1,6 %.

Les deux chantiers ne se contredisent pas. Ils optimisent les deux bouts d'une
même courbe, et **la métrique qui les compare est `J / réponse juste`**, pas
`tok/s` ni `params`.

## 5. Les trois régimes, en une figure

```
  qualité
     ▲
     │                                   ╱  régime 100 Go
     │                                 ╱    (sparsité, noyaux)
     │                        ╱──────╱
     │                 ╱────╱   régime 1 Go
     │          ╱────╱        (compromis)
     │    ╱────╱
     │ ╱──╱  régime 100 Mo
     │╱     (coûts fixes, données)
     └──────────────────────────────────────────►  RAM
       0,1 Go     1 Go      10 Go     100 Go
```

Les trois droites ne sont pas parallèles : la pente s'effondre quand les coûts
fixes dominent, puis remonte quand la sparsité devient exploitable. **Le pire
endroit où se placer est entre 200 Mo et 2 Go** : assez gros pour payer un
runtime complet, trop petit pour bénéficier de la sparsité.

## 6. Les cinq pièges arithmétiques

1. **Le KV est relu à chaque token.** Ce n'est pas seulement de la RAM, c'est
   de la bande passante. À 32 k avec GQA 2 têtes et 8 bits, 48 couches →
   0,8 Go/token rien qu'en KV. Avec MLA compressé et 4 bits → 0,15 Go. Un
   facteur 5 sur le débit total, pour la même architecture.
2. **La tête de sortie est relue à chaque token.** 131 k × 2 048 à 6 bits =
   0,20 Go/token, soit 10 % du budget. Diviser le vocabulaire par deux divise
   ce coût par deux.
3. **Le routeur grossit avec le nombre d'experts.** 64 couches × 2 048 × 1 536
   à 8 bits = 0,25 Go/token, 19 % du budget. Un routeur de rang 256 ramène
   cela à 0,05 Go. Personne ne le mentionne dans les papiers MoE.
4. **La taxe ternaire n'est pas gratuite.** 1,58 bit théorique devient 1,7 bit
   effectif avec les échelles de groupe et l'alignement. Sur 360 B de
   paramètres, c'est 5 Go de différence.
5. **`GiB` ≠ `Go`.** Un modèle « 60,8 GiB » pèse 65,3 Go. Sur un budget de
   96 Go, l'écart vaut 5 % de marge.

## 7. Comment utiliser la loi pour concevoir

Dans l'ordre, pour un budget `M` et une cible de débit `v` :

1. Calculer le budget de poids : `M − runtime − KV(T) − buffers`.
2. Choisir `b` (1,7 si la qualité tient, 4 sinon).
3. En déduire `P_max = 8 · budget_poids / b`.
4. Choisir `ρ` pour tenir `v` : `ρ ≤ v · budget_poids / (BW · P_max)`.
5. En déduire `P_actif = ρ · P_max`, puis le nombre d'experts `E = P_actif / (top_k · 3·d·d_exp)`.
6. Vérifier le seuil du noyau : `GOPS ≥ 16 · BW / b`.
7. Vérifier la mémoire du KV à la longueur de contexte cible.
8. Seulement là, choisir la profondeur et `d_model`.

`tools/hecaton_budget.py law` fait les étapes 1-5 automatiquement ;
`tools/hecaton_budget.py sensitivity` fait l'étape 6.

## 8. Limites de la loi

- **Elle ne dit rien de la qualité.** `P` et `b` sont des proxys ; la relation
  qualité-paramètres-bits n'est calibrée nulle part au-delà de 3 B.
- **Elle suppose un noyau idéal.** Les GOP/s réels dépendent du packing, des
  échelles, de la vectorisation et de la mémoire.
- **Elle ignore la latence.** 49 tok/s suffisent pour lire ; 15 s de prefill
  pour 1 000 tokens ne suffisent pas pour interagir (voir §3 du document
  noyaux).
- **Elle suppose une architecture connue.** Une architecture récurrente ou à
  mémoire externe change les équations (1) et (3).
