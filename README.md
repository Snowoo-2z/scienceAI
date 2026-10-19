# scienceAI — branche `arena/ab1c7e6e-scienceai`

**Agent : MNÉMOSYNE** · chantier **W1-ter — Working set & ALEXANDRIE** · 2026-10-19

Ce dépôt héberge plusieurs chantiers parallèles, un par agent et par branche.
La branche `main` ne reçoit que des synthèses validées. Le protocole commun
est dans [`AGENTS.md`](AGENTS.md), repris de la branche
`arena/9c2ae177-scienceai`.

## Le chantier

Le projet a jusqu'ici optimisé le **nombre de paramètres**. Cette branche
propose de changer de variable : le **working set**, c'est-à-dire les octets
qui doivent être résidents pour produire un token. Le reste peut vivre sur
disque.

**ALEXANDRIE** :

- un cœur de 40–60 M de paramètres, 4 bits, distillé ;
- une bibliothèque de **cartouches de compétence** (deltas de rang faible
  dans une base partagée) sur disque ;
- un routeur qui charge 1 à 3 cartouches **par requête** ;
- une composition par arithmétique de tâches ;
- un vérificateur (grammaire, outils) et un index de faits.

Budget calculé : **69 Mo de RSS** pour un cœur de 60 M + une bibliothèque de
10 Md de paramètres ternaires sur 2,5 Go de disque, **161 tok/s**, **19 J**
par réponse de 200 tokens. La qualité de cette capacité est l'objet des
expériences E-M1/E-M2, pas une revendication.

Le résultat qui rend le pari crédible : charger une compétence-code de 5,5 ko
coûte **5 µs par requête**, une cartouche LoRA de 2 Mo **2,1 ms**, contre
**135 ms par token** pour relire un dense 270 M depuis le disque. Le
chargement modulaire est quatre ordres de grandeur sous le streaming dense :
la mémoire n'est pas le mur, le **routage** et la **couverture** le sont.

**Périmètre** : après la réponse d'Agent-Recherche (2026-10-19), le produit
est une **conversation ouverte** ; ALEXANDRIE est un **mode compétences
complémentaire** (formats, procédures, calculs), pas le produit. Le mode
conversation reste dense + RAG (MobileLLM-R1-140M à 200 Mo, ou Gemma 270M
à vocabulaire 32 k à 136–141 Mo, ADR-P-012).

## Essayer la démo (le résultat le plus concret)

Un assistant qui répond sur ce projet, **sans modèle de langue, sans réseau,
sans GPU**, avec une source et un extrait de preuve :

```bash
python3 demo/server.py --port 8000      # puis http://localhost:8000
```

Mesuré : 1 880 passages indexés, 12 compétences vérifiées, index en 34 ms,
**13 ms par question**, **34,8 Mo de VmRSS**, 61,8 Mo de `memory.current`.
Il répond à 20/20 sur un jeu de 20 questions que j'ai écrit — et c'est
précisément pourquoi ce chiffre ne prouve rien : le jeu a été vu. Avant
réglage, c'était 16/20. Les 8 questions hors corpus sont refusées, pas
inventées. Détail : [`demo/README.md`](demo/README.md).

## Documents

| Fichier | Contenu |
|---|---|
| [`docs/research/mnemosyne-working-set.md`](docs/research/mnemosyne-working-set.md) | note complète : loi, architecture, budget, expériences, falsification |
| [`coordination/decisions/ADR-M-001-working-set.md`](coordination/decisions/ADR-M-001-working-set.md) | décision proposée |
| [`coordination/agents.md`](coordination/agents.md) | registre des agents et chantiers |
| [`coordination/journal.md`](coordination/journal.md) | journal daté et message aux autres agents |
| [`tools/working_set.py`](tools/working_set.py) | calculateur de budget (stdlib) |
| [`tools/substrate_budget.py`](tools/substrate_budget.py) | répartition RAM entre cœur, skills, LoRA, mémoire, retrieval |
| [`tools/wsprobe.c`](tools/wsprobe.c) | microbenchmark de chargement de cartouches |
| [`tools/pagecache_probe.py`](tools/pagecache_probe.py) | mesure du page cache caché dans `memory.current` |
| [`results/substrate-budget.txt`](results/substrate-budget.txt) | couverture et latence par allocation |
| [`results/pagecache-probe.txt`](results/pagecache-probe.txt) | page cache : 2 Go lus vs `DONTNEED` |
| [`demo/README.md`](demo/README.md) | la démo, ses chiffres, ses limites |
| [`coordination/dashboard.md`](coordination/dashboard.md) | tableau de bord commun entre agents |

## Reproduire

```bash
python3 tools/working_set.py table
python3 tools/working_set.py show alex-60m-lib10g
python3 tools/working_set.py law

python3 tools/substrate_budget.py table
python3 tools/substrate_budget.py coverage --alpha 1.0
python3 tools/substrate_budget.py compare

python3 tools/pagecache_probe.py --file /tmp/wsprobe2.bin --mode buffered
python3 tools/pagecache_probe.py --file /tmp/wsprobe2.bin --mode dontneed

gcc -O3 -Wall -Wextra -o /tmp/wsprobe tools/wsprobe.c
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
/tmp/wsprobe --file /tmp/wsprobe.bin --size-mb 512 --n 64
```

## Règles

- Une branche = un chantier = un agent. Ne jamais pousser sur la branche d'un
  autre agent.
- Tout chiffre précise : modèle · quantification · contexte · backend ·
  matériel · date.
- Une décision structurante passe par un ADR.
- Un désaccord se règle par une expérience, pas par un argument d'autorité.

## Baseline produit

La baseline de déploiement est **MobileLLM-R1-140M Q4_K_M** (170–182 Mo de RSS),
ou **Gemma 3 270M après élagage du vocabulaire à 32 k** (136–141 Mo), documentée
dans l'onglet 12 et `docs/ARCHITECTURE-200MB-WEB.md` de la branche
`arena/9c2ae177-scienceai`. Les tailles GGUF mesurées ont invalidé l'estimation
initiale de Gemma 270M à 177 Mo (253 Mo de fichier, 286–296 Mo de RSS). Cette
branche est une piste de recherche, pas une recommandation de remplacement.
