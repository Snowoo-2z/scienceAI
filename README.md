# scienceAI — Project LLM

Objectif : concevoir et **déployer** un LLM **généraliste utile sous ~200 Mo de RAM** (avec 100 Mo
comme aspiration), **entraînable sous 1 Go**, **sans gros moyens de calcul**, et **économe en
énergie**, servi par un **petit PC** à travers un **site web**.

Ce dépôt est un espace de recherche partagé entre plusieurs agents travaillant sur des branches
différentes. Le canal de synchronisation principal est le Google Doc **« Project LLM »** ; le dépôt
en est la version vérifiable.

## Démarrage rapide

```bash
# Voir les architectures candidates et leur budget mémoire
python tools/llm_budget.py table

# Détail d'une configuration
python tools/llm_budget.py show gemma3-270m-qat-8k

# Dimensionner un serveur web (slots, RSS, latence, énergie)
python tools/llm_budget.py serve gemma3-270m-qat-8k --parallel 4 --ctx-per-slot 2048

# Énergie réellement payée au mur
python tools/llm_budget.py wall gemma3-270m-qat-8k --device-watts 15

# Budget d'entraînement (mémoire, jours, coût)
python tools/llm_budget.py train gemma3-270m-qat-8k --tokens 2.0e7
```

Aucune dépendance : Python 3.9+ suffit.

## Document de travail partagé

Google Doc **« Project LLM »** (13 onglets, canal de coordination des agents) :
https://docs.google.com/document/d/13-LF4vy1dyeh3JyWM8hHw5aoKJz1fpf5OeaUXn7CzjA/edit

## Documents

| Fichier | Contenu |
|---|---|
| [`docs/ARCHITECTURE-200MB-WEB.md`](docs/ARCHITECTURE-200MB-WEB.md) | **Recommandation v0.2** : Gemma 3 270M, QLoRA ≤ 1 Go, déploiement web sur petits PC |
| [`docs/RECHERCHE-LLM-100MB.md`](docs/RECHERCHE-LLM-100MB.md) | Rapport de recherche v0.1 (état de l'art, leviers, arithmétique, plan) |
| [`AGENTS.md`](AGENTS.md) | Protocole de collaboration entre agents |
| [`coordination/agents.md`](coordination/agents.md) | Registre des agents et des chantiers |
| [`coordination/decisions.md`](coordination/decisions.md) | Journal des décisions (ADR) |
| [`coordination/journal.md`](coordination/journal.md) | Journal chronologique des contributions |

## Les contraintes

1. **Inférence ≤ ~200 Mo de RAM** (poids + KV cache + runtime), 100 Mo en aspiration.
2. **Entraînement ≤ 1 Go de RAM** et **pas des mois**.
3. **Pas d'énorme puissance de calcul**, énergie minimale.
4. **Généraliste**, pas spécialisé.
5. **Livraison = site web** servi par des petits PC.

## Résultat principal (v0.2)

- **200 Mo suffisent pour un généraliste** : Gemma 3 270M en QAT INT4 tient dans **177 Mo de RSS**
  avec 8 k de contexte partagé et 4 slots.
- **1 Go suffit pour adapter, pas pour pré-entraîner** : QLoRA 5–20 M tokens, quelques heures de GPU,
  1–2 $. Le pré-entraînement from scratch plafonne à ~60 M de paramètres et donne un modèle inférieur.
- **Déploiement** : un mini-PC N100, `llama-server`, `--parallel 4`, ~9 tok/s par utilisateur,
  ~8 €/an d'électricité à 1 000 requêtes/jour.
- **Qualité** : celle d'un petit assistant. Le produit doit être bâti autour du **RAG** et de
  réponses courtes, pas présenté comme un ChatGPT.
- **Seule piste de recherche utile** : distillation 270 M → 60 M pour repasser sous 100 Mo (E6).

### Résultat historique (v0.1, hypothèse 100 Mo)

- Un généraliste sous 100 Mo n'existe pas ; un spécialisé de 60–85 M en 4-bit QAT tient dans ~78 Mo.
- Levier principal : **la donnée et la distillation**, pas la compression.
- À cette échelle, **l'énergie est dominée par la plateforme** (`E ≈ P / débit`).

## Licence / usage

Recherche interne. Les chiffres et affirmations doivent être reproductibles ou sourcés.
