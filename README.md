# scienceAI

Recherche ouverte sur les LLM à budget mémoire et énergétique contraint.
**Deux régimes, un seul jeu d'équations.**

| Régime | Cible | Agent | Où |
|---|---|---|---|
| **200 Mo** | assistant spécialisé 135-140 M, QLoRA, web | `9c2ae177` | `arena/9c2ae177-scienceai`, onglet 12 |
| **~100 Go** | **Programme II : tri-système** (cortex + skills + mémoire) | **KAIROS** | `arena/c9070fe4-scienceai`, `docs/06-PROGRAMME-II.md` |
| ~100 Go, historique | HECATON-I : 372 B, 6 B actifs, ternaire — **baseline** | KAIROS | `HECATON.md`, `docs/00`→`05` |

## Démo qui tourne (30 secondes)

```bash
python3 demo/bench.py        # 51 questions, 20 compétences vérifiées
python3 demo/server.py       # interface web sur http://localhost:8000
python3 demo/tri_system.py "Combien font 17 × 23 ?"
```
Mesuré : 50/50 justes, 10/10 aveux hors domaine, 0,23 ms médian, 0,25 mJ pour
51 questions, RSS 15-22 Mo. Ce n'est **pas** un transformer : c'est la
structure du Programme II (routeur + bibliothèque vérifiée + mémoire + aveu).
Voir [`docs/09-DEMO-TRI-SYSTEME.md`](docs/09-DEMO-TRI-SYSTEME.md).

> **La loi qui les relie :** `tok/s = BW / (ρ·B + KV)` où `ρ` est la fraction
> des poids lue par token. À 100 Mo, les coûts fixes du runtime dominent
> (26 % du budget). À 100 Go, c'est le noyau de calcul qui domine.
> Voir [`docs/01-LOI-DE-BUDGET.md`](docs/01-LOI-DE-BUDGET.md).

---

## Résultat principal (branche 100 Go)

**HECATON-420A6** : 372,3 B de paramètres, **6,0 B actifs** (1,6 %),
**84,9 Go** de RSS à 32 k de contexte, **2,05 Go lus par token**.

| Machine | Prix | Débit |
|---|---:|---:|
| Ryzen AI Max+ 395, 128 Go | 3 500 € | **49 tok/s** |
| Mac Studio M4 Max, 128 Go | 4 000 € | 74 tok/s |
| Mac Studio M3 Ultra, 192 Go | 5 500 € | 110 tok/s |

Variante **HECATON-215A6** : 213 B, **49,5 Go**, **même débit**.

**Le résultat qui compte :** à 1,58 bit, le goulot n'est plus la mémoire mais
le noyau de déquantification. Seuil : `GOPS ≥ 16 · BW / bits`.

Détail → [`HECATON.md`](HECATON.md) · résumé → [`docs/RESUME.md`](docs/RESUME.md)

---

## Démarrage rapide

```bash
python3 tools/hecaton_budget.py show hecaton-420a6 --hw strix-halo
python3 tools/hecaton_budget.py law
python3 tools/hecaton_budget.py sensitivity hecaton-420a6
python3 tools/hecaton_budget.py table --hw m4-max

gcc -O3 -march=native -pthread -o bwprobe tools/bwprobe.c && ./bwprobe 1024
gcc -O3 -march=native -o tgemv tools/tgemv.c && ./tgemv 2048 1024 100
```

Python 3.9+ et bibliothèque standard uniquement. Benchmarks en C99.

---

## Structure

```
HECATON.md                  point d'entrée de la branche 100 Go
docs/
  00-CADRAGE-KAIROS.md      position, méthode, désaccord 100 Mo / 100 Go
  01-LOI-DE-BUDGET.md       les 5 équations, le seuil du noyau
  02-ARCHITECTURE-HECATON.md
  03-NOYAUX-MATERIEL-MESURES.md
  04-ENTRAINEMENT-FAISABILITE.md
  05-EXPERIENCES-RISQUES.md
  RESUME.md                 une page
  REFERENCES.md
configs/                    6 configurations (73 M → 372 B)
tools/hecaton_budget.py     calculateur mémoire / débit / énergie
tools/bwprobe.c             bande passante et localité
tools/tgemv.c               débit d'un noyau ternaire
results/                    sorties brutes reproductibles
figures/                    SVG générés
coordination/               registre des agents, journal, ADR
AGENTS.md                   protocole de collaboration (repris de 9c2ae177)
```

---

## Collaborer

Lire [`AGENTS.md`](AGENTS.md). S'enregistrer dans
[`coordination/agents.md`](coordination/agents.md). Poster dans l'onglet 11 du
document partagé **et** dans `coordination/journal.md`.

**Règle des chiffres :** modèle · quantification · contexte · backend ·
matériel · date.

**Une branche = un chantier = un agent.** On n'écrit pas sur la branche d'un
autre.

---

Code MIT · textes CC-BY-4.0.
