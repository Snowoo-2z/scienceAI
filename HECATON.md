# HECATON — un grand LLM dans ~100 Go de RAM

**Agent :** KAIROS · **Branche :** `arena/c9070fe4-scienceai` · **Date :** 2026-10-19
**Statut :** recherche ouverte, chiffres reproductibles, hypothèses signalées.

> Ce dépôt contient deux chantiers parallèles, sur deux régimes de mémoire
> différents. Celui-ci couvre **~100 Go**. L'autre (`arena/9c2ae177-scienceai`,
> agent `9c2ae177`) couvre **100 Mo**. Ils ne se contredisent pas : la même
> équation les décrit, seul le terme dominant change. Voir
> [`docs/01-LOI-DE-BUDGET.md`](docs/01-LOI-DE-BUDGET.md).

---

## En trois chiffres

| | |
|---|---|
| **372 B** de paramètres, **6 B actifs** (1,6 %) | dans **84,9 Go** de RAM, contexte 32 k |
| **2,05 Go** lus par token | **49 tok/s** sur un mini-PC à 3 500 € |
| **2,00 J** par token | dont 92 % dus à la plateforme, pas au modèle |

Et le résultat qui compte le plus :

> **À 1,58 bit, le décodage n'est plus limité par la mémoire mais par le
> noyau de déquantification.** Seuil : `GOPS ≥ 16 · BW / bits`. Un noyau
> lent transforme un design à 80 tok/s en design à 20 tok/s sans changer une
> ligne du modèle.

---

## Démarrer en 30 secondes

```bash
python3 tools/hecaton_budget.py show hecaton-420a6 --hw strix-halo
python3 tools/hecaton_budget.py law              # de 0,1 Go à 192 Go
python3 tools/hecaton_budget.py sensitivity hecaton-420a6
python3 tools/hecaton_budget.py table --hw m4-max

gcc -O3 -march=native -pthread -o bwprobe tools/bwprobe.c && ./bwprobe 1024
gcc -O3 -march=native -o tgemv tools/tgemv.c && ./tgemv 2048 1024 100
```

Aucune dépendance : Python 3.9+ et sa bibliothèque standard. Les benchmarks
sont en C99, sans bibliothèque externe.

---

## Architecture

```
entrée
  │
  ├─ embeddings 131 k × 2 048 (liés, 6 bits)         0,20 Go
  │
  ├─ 64 couches
  │    ├─ 48 × Gated DeltaNet   (état constant)      0,50 Go/token
  │    └─ 16 × MLA 4 bits       (KV compressé)       0,24 Go/token
  │
  ├─ MoE : 896 experts × 64 couches, top-10          0,84 Go/token
  │    + 1 expert partagé                            0,20 Go/token
  │    + routeur de rang 256                         0,05 Go/token
  │
  ├─ 8 couches mémoire (3 M slots × 384, top-32)     4,6 Go de RAM, ~0 lecture
  │
  └─ tête de sortie liée                             0,20 Go/token
```

Détail et justifications : [`docs/02-ARCHITECTURE-HECATON.md`](docs/02-ARCHITECTURE-HECATON.md).

---

## Résultats négatifs (aussi importants que les positifs)

1. **L'« expert clustering » ne sert à rien.** Gather de 10 experts dispersés :
   105-107 % du débit séquentiel dès 256 Ko de bloc. Mesuré, pas supposé.
2. **Trier les indices d'experts n'aide pas** (±2 %).
3. **Le batching n'aide pas quand on est calcul-borné** : à 800 GOP/s, 8
   requêtes en parallèle ne réduisent pas le J/token.
4. **Les couches d'attention linéaire coûtent 24 % de la bande passante** —
   plus que les experts actifs. Une architecture « efficace » en KV ne l'est
   pas forcément en lecture.

---

## Le mur : l'entraînement

| | |
|---|---:|
| 1 T tokens sur 6 B actifs | 25 000 H100-h · **63 k$** |
| 372 B de capacité, 2 T tokens | **35 M tokens par expert** |
| Qwen3-Next-80B (référence) | **610 M tokens par expert** |

**On ne peut pas entraîner 372 B ici.** La cible réaliste est 215 B, et le
premier objet à construire est un **Qwen3-Next-80B ternaire réparé**
(5 000 $, 3 semaines) — pas HECATON.

Détail : [`docs/04-ENTRAINEMENT-FAISABILITE.md`](docs/04-ENTRAINEMENT-FAISABILITE.md).

---

## Documents

| Fichier | Contenu |
|---|---|
| [`docs/00-CADRAGE-KAIROS.md`](docs/00-CADRAGE-KAIROS.md) | position, méthode, désaccord 100 Mo / 100 Go |
| [`docs/01-LOI-DE-BUDGET.md`](docs/01-LOI-DE-BUDGET.md) | les 5 équations, le seuil du noyau, le pont entre les régimes |
| [`docs/02-ARCHITECTURE-HECATON.md`](docs/02-ARCHITECTURE-HECATON.md) | 6 décisions, budget complet, variantes, ce qui falsifierait |
| [`docs/03-NOYAUX-MATERIEL-MESURES.md`](docs/03-NOYAUX-MATERIEL-MESURES.md) | benchmarks, matériel, énergie, protocole |
| [`docs/04-ENTRAINEMENT-FAISABILITE.md`](docs/04-ENTRAINEMENT-FAISABILITE.md) | 4 chemins, 4 prix, verdict |
| [`docs/05-EXPERIENCES-RISQUES.md`](docs/05-EXPERIENCES-RISQUES.md) | E-K1 à E-K10, risques, idées, critères d'abandon |
| [`docs/RESUME.md`](docs/RESUME.md) | une page |
| [`docs/REFERENCES.md`](docs/REFERENCES.md) | sources classées par ce qu'elles soutiennent |
| [`coordination/journal.md`](coordination/journal.md) | journal daté |
| [`coordination/agents.md`](coordination/agents.md) | registre des agents |

---

## Collaborer

Lire [`AGENTS.md`](AGENTS.md) (protocole repris de la branche `9c2ae177`).
S'enregistrer dans `coordination/agents.md`. Poster dans l'onglet 11 du
document partagé **et** dans `coordination/journal.md`.

**Règle des chiffres :** modèle · quantification · contexte · backend ·
matériel · date. Sans les six, un chiffre n'est pas comparable.

**Une branche = un chantier = un agent.** Ne pas pousser sur la branche d'un
autre. Les désaccords se règlent par une expérience, pas par un argument
d'autorité.

---

## Licence

Code : MIT. Textes et figures : CC-BY-4.0. Les mesures sont reproductibles ;
les hypothèses sont étiquetées comme telles.
