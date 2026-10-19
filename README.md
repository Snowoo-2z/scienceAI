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

# Détail d'une configuration (tailles GGUF mesurées)
python tools/llm_budget.py show gemma3-270m-q4emb-8k

# Dimensionner un serveur web (slots, RSS, latence, énergie)
python tools/llm_budget.py serve gemma3-270m-q4emb-8k --parallel 4 --ctx-per-slot 2048

# Énergie réellement payée au mur
python tools/llm_budget.py wall gemma3-270m-q4emb-8k --device-watts 15

# Budget d'entraînement (mémoire, jours, coût)
python tools/llm_budget.py train gemma3-270m-q4emb-8k --tokens 2.0e7
```

Aucune dépendance : Python 3.9+ suffit.

## Démonstrateur qui tourne (2026-10-19)

**NanoFR** : un transformer « architecture LLaMA » écrit à la main en numpy,
entraîné sur 16 Mo de littérature française (CC-BY), exporté en GGUF, quantifié,
et servi par llama.cpp dans une page web.

```bash
python3 tools/prepare_corpus.py          # corpus (isosceles + ELTeC-fra)
python3 tools/train_tokenizer.py         # BPE 8 192 pièces
python3 tools/tokenize_corpus.py
python3 tools/nanofr_train.py --steps 2400 --batch 8   # ~640 Mo de pic RSS
python3 tools/export_gguf.py data/nanofr_ckpt.npz data/nanofr_f32.gguf
llama-quantize data/nanofr_f32.gguf data/nanofr_q8_0.gguf Q8_0
llama-server -m data/nanofr_q8_0.gguf -c 256 -t 2 --port 8081 &
python3 app/server.py --port 8000
```

Résultats mesurés : 6,52 M de paramètres, 4,92 M de tokens vus en 60 min sous
**824 Mo de pic RSS**, perplexité de validation 81,8 (1,67 bit/caractère).
En inférence : **28,9 Mo de RSS** en Q8 pour 1 358–1 922 tok/s sur deux cœurs,
application web comprise (20,2 Mo).
Détail, méthode et limites : `docs/NANOFR.md`.

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

## Résultat principal (v0.3)

- **Un généraliste tient dans 200 Mo.** Gemma 3 270M QAT INT4 avec embeddings quantifiés en 4 bits :
  **183 Mo de RSS à 2 k de contexte, 193 Mo à 8 k**.
- **La condition est la quantification des embeddings.** Les GGUF Q4_K_M standard les gardent à
  8 bits : 262 144 × 640 = 167,8 M paramètres, 253 Mo de fichier, 286–296 Mo de RSS. En Q4_0 sur
  tous les tenseurs, le fichier tombe à ~150 Mo.
- **Pourquoi ça marche** : la QAT officielle est entraînée à supporter l'INT4 ; et Gemma 3 a une
  seule tête KV avec 15 couches sur 18 à fenêtre glissante, soit 512 octets de KV par token et par
  couche — 16,5 Mo pour 8 k de contexte.
- **1 Go suffit pour adapter, pas pour pré-entraîner** : QLoRA NF4 (0,16 Go), 5–20 M tokens,
  quelques heures de GPU, 1–2 $.
- **Déploiement** : un mini-PC N100, `llama-server`, `--ctx-size 8192 --parallel 4`,
  ~9 tok/s par visiteur, ~8 €/an d'électricité pour 1 000 requêtes/jour.
- **Qualité** : celle d'un petit assistant. Le produit doit être bâti autour du **RAG** et de
  réponses courtes, pas présenté comme un ChatGPT.

### Résultat historique (v0.1, hypothèse 100 Mo)

- Un généraliste sous 100 Mo n'existe pas ; un spécialisé de 60–85 M en 4-bit QAT tient dans ~78 Mo.
- Levier principal : **la donnée et la distillation**, pas la compression.
- À cette échelle, **l'énergie est dominée par la plateforme** (`E ≈ P / débit`).

## Licence / usage

Recherche interne. Les chiffres et affirmations doivent être reproductibles ou sourcés.
