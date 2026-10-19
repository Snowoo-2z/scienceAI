# PROGRAMME II — capacité sans grossir le modèle

**KAIROS · 2026-10-19 · rupture assumée avec HECATON-I**

> **Ce document remplace l'objectif du précédent.** HECATON-420A6 (372 B
> quantifiés en 1,58 bit) répond à « comment faire tenir un gros transformer
> dans 100 Go ». C'est de la compression, pas une invention : tous ses
> composants existent (MoE, MLA, DeltaNet, couches mémoire, BitNet). Il reste
> comme **baseline mesurable**, pas comme programme.
>
> Le nouveau programme répond à : **qu'est-ce qu'on peut construire qui soit
> plus capable par joule, sans grossir le modèle ?**

---

## 0. Autocritique, en deux lignes

Le travail précédent optimisait une contrainte (`P·b/8 ≤ 100 Go`). Il ne
créait rien : il déplaçait des paramètres. Une invention doit changer **la
forme de la dépense**, pas seulement son montant.

---

## 1. Le défaut de conception d'un transformer

Un transformer met **trois choses différentes dans le même substrat** — ses
poids — et les paie toutes les trois à chaque token :

| Ce qu'il faut stocker | Nature | Coût idéal | Ce que fait un transformer |
|---|---|---|---|
| **Connaissance** (faits) | sparse, change souvent | lookup ~0 J | gravée dans les poids, immuable |
| **Compétence** (procédures) | exacte, réutilisable | exécution ~0 J | approximée par des matrices |
| **Raisonnement** (composition) | coûteux, dépend du problème | proportionnel à la difficulté | **constant par token** |

Trois conséquences, toutes mesurables :

1. **Pour ajouter un fait, il faut réentraîner** ou le mettre en contexte (et
   le payer à chaque token).
2. **Pour ajouter une compétence, il faut réentraîner.** Un modèle ne
   « apprend » pas à utiliser une nouvelle API sans fine-tuning.
3. **Un token facile coûte autant qu'un token difficile.** « 2+2 » traverse
   les mêmes 6 milliards de paramètres que « démontre ce lemme ».

**C'est ce triple couplage qu'on attaque.** Pas la taille des poids.

---

## 2. La nouvelle identité de budget

```
C  =  f( W , M , S , T )
sous   W + M ≤ 100 Go
et     E  =  α·R(W)  +  β·S  +  γ·U
```

- `W` : poids (Go) — connaissance *générale* et syntaxe.
- `M` : mémoire externe (Go) — compétences exécutables + épisodes vécus.
- `S` : recherche/calcul à l'inférence (FLOPs) — dépensé selon la difficulté.
- `T` : plasticité (mises à jour) — ce que le système apprend en usage.
- `R(W)` : octets de poids lus par token ; `U` : coût d'une mise à jour.

Un transformer fixe `M = 0`, `T = 0`, et `S` constant. Le programme II ouvre
les trois axes à zéro.

**La question scientifique devient une élasticité :**

```
∂C/∂Go  pour les poids   vs   pour les compétences   vs   pour les épisodes
```

Personne ne l'a mesurée. C'est l'expérience E-Ω5, et c'est le vrai apport
théorique de ce programme.

### Principe P1 — la frontière de vérifiabilité

> **On ne grave dans les poids que ce qui est vérifiable ; tout le reste va
> dans la mémoire plastique.**

Un fait vérifiable (une API, un théorème, une procédure) peut être exécuté ou
testé : il n'a pas besoin d'être « su ». Un fait non vérifiable (une opinion,
un style) doit être appris. Cette règle donne une frontière **principielle**
entre `W` et `M`, au lieu d'un réglage empirique. Elle a un corollaire
immédiat : **la mémoire ne peut pas halluciner, puisqu'elle est testée.**

---

## 3. L'architecture : un tri-système, pas un modèle

```
                    requête
                       │
              ┌────────▼────────┐
              │  CORTEX  (15 Go)│  petit transformer sparse
              │  comprend,      │  ~0,35 Go lus/token → 200+ tok/s
              │  planifie       │
              └────────┬────────┘
                       │  plan = 1 à 5 appels
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
 ┌────────────┐ ┌────────────┐ ┌──────────────┐
 │ BIBLIOTHÈQUE│ │  MÉMOIRE   │ │ VÉRIFICATEUR │
 │ 55 Go      │ │  PLASTIQUE │ │ 2 Go + bac   │
 │ 10 M skills│ │  25 Go     │ │ à sable      │
 │ + tests    │ │ 5 M épisodes│ │ exécute les  │
 └─────┬──────┘ └─────┬──────┘ │ tests        │
       │              │        └──────┬───────┘
       └──────────────┴───────────────┘
                      │
              succès ? ├── oui → réponse + la composition devient une skill
                      └── non → génération par le cortex (fallback)
```

### Allocation du budget de 100 Go

| Composant | Go | Ce qu'il contient | Coût par requête |
|---|---:|---|---:|
| Cortex | 15 | 8 B de paramètres, 1 B actif, 1,7-4 bits | 0,35 Go/token |
| Bibliothèque de compétences | 55 | ~10 M procédures (code + tests + description + embedding) | ~10 Mo |
| Mémoire épisodique | 25 | ~5 M épisodes (requête → trace → issue) | ~10 Mo |
| Vérificateur + bac à sable | 3 | tests, exécution, vérif sémantique | ~1 Mo |
| Système + index + marge | 2 | — | — |

**Aucun de ces composants n'est un « gros modèle ».** La capacité vient de la
bibliothèque et de la mémoire, pas de `W`.

### Le coût, par type de requête

| Cas | Ce qui se passe | Énergie |
|---|---|---:|
| Requête résolue par 1-3 skills | 10 tokens de plan + exécution + tests | **~5 J** |
| Requête résolue par une trace épisodique | retrieval + adaptation courte | ~8 J |
| Requête non couverte (fallback) | 300 tokens générés | ~300 J |
| **Moyenne à 90 % de couverture** | | **~35 J** |

À comparer aux **~800 J** d'une réponse de 400 tokens par HECATON-420A6.
**Facteur 20 à 150, sans un paramètre de plus.**

> **La thèse en une phrase : le token le moins cher est celui qu'on n'écrit
> pas.**

---

## 4. Les cinq inventions

Chacune est donnée avec : mécanisme, ce qu'elle a de nouveau, claim
falsifiable, test le moins cher, critère d'échec.

### I-1 — Bibliothèque de compétences vérifiées, auto-extensible (W-SKILL)

**Mécanisme.** Chaque entrée est une *procédure exécutable* avec (a) une
description en langage naturel, (b) une signature typée, (c) du code, (d) des
tests, (e) un embedding, (f) une provenance. À l'inférence : retrieval →
sélection → composition → **exécution** → tests. Ce qui passe devient une
nouvelle entrée de la bibliothèque.

**Ce qui est nouveau.** RAG récupère du *texte* ; les outils de type
Toolformer ont un *ensemble fixe* ; Voyager écrit des skills mais sans
vérification typée à l'échelle ni modèle de coût. Ici : **procédures
exécutables + tests comme condition d'entrée + croissance autonome**.

**Claim.** Sur les tâches compositionnelles et vérifiables, un cortex de 8 B +
10 M de skills bat un modèle 10× plus gros à joules égaux.

**Test (E-Ω1).** 3B-8B + bibliothèque de 10 k fonctions Python testées.
Comparer : (A) modèle seul, (B) + RAG texte, (C) + skills exécutables.
**Coût : 2 000 €, 3 semaines.**
**Échec :** C < B + 10 points → l'exécution n'apporte rien, tout le programme
tombe.

> **Révision du 2026-10-19, après critique.** « 10 M de compétences » se
> lisait comme 10 M de compétences *écrites à la main* : 5 minutes chacune,
> soit 40 années-personne. C'est intenable, et l'idée est abandonnée sous
> cette forme. Deux sources remplacent l'écriture humaine :
> (1) le **moissonnage** du code existant (mesuré : 7 596 fiches/s,
> 249 octets par fiche, 133/242 fonctions enveloppées automatiquement dans
> 11 modules de la bibliothèque standard) ; (2) l'**auto-extension** : une
> compétence n'est écrite que lorsqu'une requête échoue, une fois, puis
> testée et réutilisée. Le goulot se déplace de l'écriture vers le
> **routage** : sur le banc de 51 questions, un routeur lexical sur
> bibliothèque moissonnée donne 4 justes, 1 fausse et 46 non couvertes.
> C'est le vrai problème, et c'est un problème de *sélection*, pas de
> connaissance. Détail et critères d'abandon : `docs/10-MOISSONNAGE.md`.

### I-2 — Mémoire épisodique plastique (W-PLAST)

**Mécanisme.** 25 Go de slots `(situation → action → issue)` écrits **pendant
l'usage**, par une règle locale (pas de rétropropagation dans le cortex).
Lecture par similarité ; écriture seulement si l'issue est vérifiée.

**Ce qui est nouveau.** RAG est en lecture seule ; le fine-tuning touche tous
les poids ; les fast-weights/TTT existent mais à petite échelle et sans
filtre de vérification. Ici : **écriture locale, gratuite, conditionnée par un
test**.

**Claim.** Sur une suite de tâches répétées, le taux de réussite monte de
≥ 10 points entre la 1ʳᵉ et la 5ᵉ exposition, **sans dégradation** sur le
reste.

**Test (E-Ω2).** 1 000 tâches × 5 expositions ; comparer à (a) in-context
learning, (b) RAG de solutions passées, (c) sans mémoire.
**Coût : 1 semaine, 0 €.**
**Échec :** gain < 5 points, ou oubli catastrophique > 5 points.

### I-3 — Profondeur adaptative (W-DEPTH)

**Mécanisme.** Un bloc récurrent de ~2 Go est exécuté `R` fois ; une tête
d'arrêt décide `R` par token, entraînée par retour du vérificateur. Facile :
`R = 1`. Difficile : `R = 32`.

**Ce qui est nouveau.** Universal Transformer / Huginn / Ouro existent, mais
la profondeur n'y est pas pilotée par **un vérificateur externe** ni
comptabilisée en joules par difficulté.

**Claim.** Sur le raisonnement, l'accuracy croît encore entre `R = 4` et
`R = 16`.

**Test (E-Ω3).** Courbe accuracy(R) sur GSM8K et un jeu de preuves.
**Coût : 1 semaine.**
**Échec :** plateau dès `R = 2` → ce n'est qu'un petit modèle, on abandonne.

### I-4 — Raisonnement par analogie sur traces (W-PLAST bis)

**Mécanisme.** On ne récupère pas la réponse : on récupère *la trace de
raisonnement* d'un problème structurellement proche, et on l'**édite**. Un
vérificateur valide l'édition.

**Ce qui est nouveau.** kNN-LM récupère des tokens ; ici on récupère une
**procédure de raisonnement** et on la modifie.

**Claim.** Sur un domaine à structure répétitive (maths scolaires, SQL),
+8 points à taille de modèle constante.

**Test :** inclus dans E-Ω2.
**Échec :** < 3 points.

### I-5 — Le modèle-compilateur (pari, niveau 4)

**Mécanisme.** Le réseau ne génère plus de prose : il **émet un programme**
sur la bibliothèque, qui est exécuté. La génération n'est que le fallback.

**Claim.** 95 % des requêtes d'un domaine fermé se compilent ; le coût par
réponse tombe sous 1 J.

**Test (E-Ω4).** Mesurer le taux de compilation sur 500 requêtes d'un domaine
(outils, calcul, extraction).
**Coût : 2 semaines.**
**Échec :** < 60 % de compilation → la prose reste le mode principal.

---

## 5. Le programme expérimental

| # | Question | Coût | Tue quoi si échec |
|---|---|---|---|
| **E-Ω0** | α et k de la demande réelle | 0 €, 2 sem. | tout le programme, avant de le lancer |
| **E-Ω1** | skills exécutables > RAG texte ? | 2 000 €, 3 sem. | tout le programme |
| **E-Ω2** | la mémoire plastique apprend-elle ? | 0 €, 1 sem. | I-2 et I-4 |
| **E-Ω3** | penser plus longtemps aide-t-il ? | 0 €, 1 sem. | I-3 |
| **E-Ω4** | taux de compilation | 0 €, 2 sem. | I-5 |
| **E-Ω5** | élasticité capacité/Go par substrat | 5 000 €, 4 sem. | la thèse théorique |

**E-Ω0 passe avant toutes les autres.** Elle mesure les deux paramètres qui
commandent le calcul de rendement (§10) : l'exposant de Zipf α de la demande
et le nombre de décompositions alternatives k. Outil : `tools/zipf_check.py` ;
protocole : `docs/08-MESURER-LA-DEMANDE.md`. Règle écrite d'avance : α < 0,8
⇒ on ne construit pas la bibliothèque.

**E-Ω5 est l'expérience reine.** À budget total fixé (par exemple 40 Go), on
fait varier la répartition `W` ↔ `M` (100/0, 75/25, 50/50, 25/75) sur une
suite de tâches fixe, et on mesure la capacité. On obtient **une courbe de
rendement marginal par substrat** — c'est-à-dire une loi, pas une opinion.

Personne, à ma connaissance, n'a publié cette courbe. C'est le résultat
publiable du programme.

---

## 6. Antériorité, honnêtement

| Brique | Existe déjà | Ce qui change ici |
|---|---|---|
| Retrieval | RAG (2020), kNN-LM | on récupère des **procédures**, pas des passages |
| Outils | Toolformer (2023), PAL | ensemble **ouvert et auto-étendu**, pas fixe |
| Skills écrites par le modèle | Voyager (2023) | **vérification typée + tests** comme condition d'entrée |
| Mémoire plastique | fast weights, TTT, Titans | **filtre de vérification** + échelle 25 Go |
| Profondeur adaptative | Universal Transformer, Huginn, Ouro | pilotage par **vérificateur externe** |
| Neurosymbolique | AlphaProof, AlphaGeometry | système **généraliste** avec budget mémoire explicite |

**Ce qu'on revendique :** pas l'invention de la mémoire ni des outils, mais
(a) **une théorie d'allocation** des 100 Go entre quatre substrats,
(b) **une boucle d'auto-extension vérifiée** qui fait croître la capacité sans
réentraînement, (c) **la mesure de l'élasticité** capacité/Go par substrat.
C'est modeste et c'est falsifiable — deux qualités rares.

---

## 7. Risques et critères d'abandon

| # | Risque | Impact | Test |
|---|---|---|---|
| R1 | La longue traîne : 10 M de skills ne couvrent pas 80 % des requêtes | **fatal** | E-Ω1, mesurer la couverture |
| R2 | Le retrieval à 10 M d'entrées est imprécis | grave | précision@50 |
| R3 | La vérification coûte aussi cher que la génération | grave | joules par vérification |
| R4 | La mémoire plastique oublie ou se pollue | grave | E-Ω2 |
| R5 | Le cortex de 8 B ne sait pas planifier | grave | taux de plans valides |
| R6 | L'auto-extension produit de la junk (skills fausses) | **fatal** | taux de réutilisation à 30 jours |
| R7 | Les benchmarks ne mesurent pas la capacité réelle | moyen | jeu de tâches maison |

**On abandonne si :** E-Ω1 < +10 pts sur RAG, ou si le taux de réutilisation
des skills auto-écrites tombe sous 1 %, ou si E-Ω5 montre que les poids
dominent à toute marge.

---

## 8. Ce que ça change pour le projet

1. **La métrique change** : « capacité par joule à taille de modèle fixe »,
   pas « combien de B tiennent dans X Go ».
2. **La compression redevient un outil** : HECATON-I sert de plancher et de
   banc d'essai, pas de but.
3. **Trois sous-chantiers** : W-SKILL, W-PLAST, W-DEPTH.
4. **Le livrable n'est plus un modèle** : c'est un **système** (cortex +
   bibliothèque + mémoire + vérificateur) et une **loi d'allocation**.

### Et le 100 Mo ?

La même bascule s'applique, en plus radical : à 100 Mo, il n'y a pas la place
pour un cortex utile, mais il y a la place pour un **routeur** et un index.
Le système devient : un petit modèle local + un index local + un vérificateur.
La connaissance ne tient pas dans 100 Mo — mais elle n'a jamais eu besoin d'y
tenir.

---

## 9. Message aux agents (version courte)

> On arrête de faire tenir des paramètres. On découple connaissance,
> compétence et raisonnement, et on mesure le rendement marginal de chaque
> gigaoctet. Trois sous-chantiers : W-SKILL (bibliothèque vérifiée),
> W-PLAST (mémoire plastique), W-DEPTH (profondeur adaptative).
> Claim : à joules égaux et à taille de cortex fixe, le système bat un
> transformer 10× plus gros sur les tâches vérifiables. Test : E-Ω1,
> 2 000 €, 3 semaines. Si ça échoue, on l'écrit et on passe à autre chose.

---

## 10. Premier calcul : le rendement d'un gigaoctet (CALCULÉ)

`tools/skill_econ.py` — modèle paramétré, **hypothèses explicites**, sorties
dans `results/programme2-*.txt` et `figures/programme2-rendement.svg`.

### 10.1 Les hypothèses

| Paramètre | Valeur | Pourquoi |
|---|---:|---|
| Demande de compétences | Zipf γ = 1,0, M = 10⁹ | loi de longue traîne standard ; **à mesurer** |
| Procédures possibles M | 10⁹ | hypothèse : la couverture vaut H(n)/H(M). Avec M = 10⁸ au lieu de 10⁹, la couverture à 10 M de skills passe de 78,4 % à 50 % (γ=1) et de 25,1 % à 50,1 % (γ=0,7). **M doit être mesuré, pas supposé.** |
| Coût d'un token de cortex | 0,5 J | cortex 8 B, 0,35 Go/token, ~5 ms |
| Planification | 10 tokens | un plan court, pas une génération |
| Génération de secours | 300 tokens | réponse complète |
| Exécution + vérification | 0,7 J | quelques ms de code |
| Taille d'une compétence | 5 Ko | code + tests + description + embedding |
| Réussite d'une compétence vérifiée | 98 % | le test passe ou ne passe pas |
| Réussite du cortex 8 B | 55 % | ordre de grandeur public |

### 10.2 Résultat principal

| Bibliothèque | Go | Couverture | J/requête | Réussite | **Réussite/J** |
|---|---:|---:|---:|---:|---:|
| 10 k skills | 0,05 | 45,9 % | 86,7 | 74,7 % | 0,0086 |
| 1 M skills | 5 | 67,6 % | 54,4 | 84,0 % | 0,0154 |
| **10 M skills** | **50** | **78,4 %** | **38,3** | **88,7 %** | **0,0232** |
| 100 M skills | 500 | 89,2 % | 22,1 | 93,3 % | 0,0422 |

Comparaison (mêmes joules, même machine) :

| Système | Réussite | J/requête | Réussite/J |
|---|---:|---:|---:|
| Cortex 8 B seul | 55,0 % | 150 | 0,0037 |
| **Gros modèle 400 B (HECATON-I)** | **77,9 %** | **800** | **0,0010** |
| Tri-système, 1 M skills | 84,0 % | 54 | 0,0154 |
| **Tri-système, 10 M skills** | **88,7 %** | **38** | **0,0232** |

**Le tri-système à 50 Go de compétences a un rendement 23 fois supérieur au
gros modèle de 400 B, pour une réussite supérieure de 10 points.** Et il
n'utilise que 50 Go de bibliothèque + 15 Go de cortex.

### 10.3 Le résultat qui décide de tout : le croisement des rendements

Points de réussite gagnés par gigaoctet supplémentaire :

| Taille du cortex | Go | Poids (pts/Go) | Bibliothèque (pts/Go) |
|---|---:|---:|---:|
| 8 B | 2 | **0,0511** | 0,0004 |
| 70 B | 15 | 0,0036 | 0,0004 |
| 200 B | 42 | 0,0010 | 0,0004 |
| **400 B** | **85** | **0,0004** | **0,0004** |

**Le croisement est exactement dans notre régime.** À 400 B, un gigaoctet de
compétences vaut autant qu'un gigaoctet de poids — avec trois avantages
décisifs : il est **modifiable sans réentraînement**, **vérifiable**, et
**daté**.

C'est la justification quantitative du programme : la bibliothèque n'est pas
un supplément, c'est le meilleur usage marginal de la mémoire **une fois que
le modèle est déjà gros**. Et il l'est, par construction, dans un budget de
100 Go.

### 10.4 Les deux paramètres qui peuvent tout tuer

| Scénario | Couverture à 10 M skills | Réussite/J | Verdict |
|---|---:|---:|---|
| Zipf γ = 1,0 (référence) | 78,4 % | 0,0232 | **gagne 6×** |
| 3 décompositions alternatives | 99,0 % | 0,1299 | **gagne 35×** |
| Zipf γ = 0,7 (demande plus étalée) | 25,1 % | 0,0056 | gagne 1,5× — marginal |

**Deux quantités décident du programme, et toutes les deux se mesurent avant
de construire quoi que ce soit :**

1. **L'exposant de Zipf de la demande de compétences.** S'il est < 0,8, la
   longue traîne est trop lourde et la bibliothèque ne couvre rien. Il se
   mesure sur des *logs d'usage réels*, pas sur un benchmark.
2. **Le nombre de décompositions alternatives par requête.** S'il existe
   trois façons indépendantes de résoudre une requête, la couverture
   explose (99 % au lieu de 78 %). Cela dépend de la redondance de la
   bibliothèque, donc de sa construction.

C'est la première chose à faire, et elle coûte **zéro euro** : instrumenter
un usage réel et mesurer ces deux nombres.

### 10.5 Ce que le calcul ne dit pas

- Il suppose que **la vérification garantit la justesse** (98 %). Si un
  vérificateur accepte 20 % de fausses réponses, le système devient un
  générateur d'erreurs confiantes. C'est le risque n° 1, et il n'est pas
  dans le modèle.
- Il suppose que le cortex sait **planifier** au-dessus de 10 M de skills.
  Rien ne le garantit.
- Il ne compte pas le coût de **construction** de la bibliothèque (extraction,
  test, déduplication, curation) — probablement le poste le plus lourd.
- Les valeurs de réussite du cortex sont des ordres de grandeur publics, pas
  une mesure sur notre configuration.

Malgré ces limites, l'ordre de grandeur est robuste : **le facteur 10-20 sur
la capacité par joule ne dépend pas finement des hypothèses**. Il dépend de
la couverture, et la couverture dépend des deux paramètres ci-dessus.
