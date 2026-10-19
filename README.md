# scienceAI — Project LLM

Objectif : concevoir un LLM **utile sous 100 Mo de RAM**, **entraînable sous 1 Go**, **sans gros
moyens de calcul**, et **économe en énergie**.

Ce dépôt est un espace de recherche partagé entre plusieurs agents travaillant sur des branches
différentes. Le canal de synchronisation principal est le Google Doc **« Project LLM »** ; le dépôt
en est la version vérifiable.

## Démarrage rapide

```bash
# Voir les architectures candidates et leur budget mémoire
python tools/llm_budget.py table

# Détail d'une configuration
python tools/llm_budget.py show nano75

# Énergie réellement payée au mur
python tools/llm_budget.py wall nano75 --device-watts 15

# Budget d'entraînement (mémoire, jours, coût)
python tools/llm_budget.py train nano75 --tokens 1.2e9 --gpu-tflops 100
```

Aucune dépendance : Python 3.9+ suffit.

## Document de travail partagé

Google Doc **« Project LLM »** (12 onglets, canal de coordination des agents) :
https://docs.google.com/document/d/13-LF4vy1dyeh3JyWM8hHw5aoKJz1fpf5OeaUXn7CzjA/edit

## Documents

| Fichier | Contenu |
|---|---|
| [`docs/RECHERCHE-LLM-100MB.md`](docs/RECHERCHE-LLM-100MB.md) | Rapport de recherche complet (état de l'art, leviers, architecture, plan) |
| [`AGENTS.md`](AGENTS.md) | Protocole de collaboration entre agents |
| [`coordination/agents.md`](coordination/agents.md) | Registre des agents et des chantiers |
| [`coordination/decisions.md`](coordination/decisions.md) | Journal des décisions (ADR) |
| [`coordination/journal.md`](coordination/journal.md) | Journal chronologique des contributions |

## Les trois contraintes

1. **Inférence ≤ ~100 Mo de RAM** (poids + KV cache + runtime).
2. **Entraînement ≤ 1 Go** et **pas des mois**.
3. **Pas d'énorme puissance de calcul**, énergie minimale.

## Résultat principal (v0.1)

- Un LLM **généraliste** sous 100 Mo n'existe pas et n'est pas atteignable.
- Un LLM **spécialisé** de **60–85 M paramètres en 4-bit QAT**, distillé, augmenté par retrieval,
  tient dans **~78 Mo** et peut être utile sur un domaine étroit.
- Une piste **ternaire 200 M** tient dans **~95 Mo** et offre 3× plus de capacité, au prix d'un
  risque d'exécution élevé.
- Le levier principal est **la donnée et la distillation**, pas la compression.
- À cette échelle, **l'énergie est dominée par la plateforme** (`E ≈ P / débit`), pas par les poids.

## Licence / usage

Recherche interne. Les chiffres et affirmations doivent être reproductibles ou sourcés.
