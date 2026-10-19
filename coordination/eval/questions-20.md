# Jeu de 20 questions — test à l'aveugle pour le démonstrateur MNÉMOSYNE

Écrit le 2026-10-19 par Agent-Recherche, **sans avoir vu les compétences** du
démonstrateur `demo/server.py`. Chaque réponse attendue est vérifiable dans le
dépôt (colonne « source »). Barème : 1 point si la réponse contient l'élément
attendu, 0 sinon. Les questions 17–20 sont **hors périmètre** : la bonne
conduite est de refuser, pas de répondre.

| # | Question | Réponse attendue | Source |
|---|---|---|---|
| 1 | Combien de RAM un généraliste de 270 M paramètres peut-il tenir, avec des embeddings en 4 bits ? | Environ 183 à 193 Mo de RSS selon le contexte (2 k–8 k), sous la cible de 200 Mo. | `docs/ARCHITECTURE-200MB-WEB.md` §3 |
| 2 | Pourquoi Gemma 3 270M ne tient-il pas dans 200 Mo en Q4_K_M standard ? | Parce que ses embeddings pèsent 168 M paramètres (262 144 × 640) et restent à ~8 bits : le fichier fait 253 Mo. | `docs/ARCHITECTURE-200MB-WEB.md` §2 |
| 3 | Quelle option de `llama-quantize` quantifie les embeddings en 4 bits ? | `--token-embedding-type q4_0`. | `docs/ARCHITECTURE-200MB-WEB.md` §3 |
| 4 | Combien d'octets de KV par token et par couche pour Gemma 3 270M ? | 512 octets (1 tête KV × head_dim 256 × 2 tenseurs × 2 octets en f16). | `docs/ARCHITECTURE-200MB-WEB.md` §4 |
| 5 | Que fait `--cache-type-k q8_0` sur la mémoire ? | Il divise par deux la taille du cache KV (environ), sans gain de vitesse garanti. | `docs/ARCHITECTURE-200MB-WEB.md` §4 |
| 6 | Quelle est la différence entre `VmRSS` et `memory.current` pour un modèle mmapé ? | `memory.current` inclut le page cache des pages du fichier ; `VmRSS` compte les pages effectivement touchées par le processus. | `docs/PROTOCOLE-EVALUATION.md` §7 (E-M0b) |
| 7 | Quelle est la seule asymétrie positive mesurée entre code et poids ? | Le code est environ trois fois plus compact que les poids pour la même couverture vérifiable (3 723 skills ≈ 20 Mo contre 10 LoRA ≈ 20 Mo). | Doc, onglet MNÉMOSYNE |
| 8 | Combien de paramètres d'embeddings Gemma 3 270M a-t-il ? | 167,8 M (262 144 × 640), soit 62 % des 270 M. | `docs/ARCHITECTURE-200MB-WEB.md` §2 |
| 9 | Quel est le plafond de RAM fixé par l'utilisateur pour l'inférence ? | 200 Mo (100 Mo visés à l'origine). | `README.md` |
| 10 | Quel est le budget d'entraînement imposé ? | 1 Go de RAM maximum, sans GPU, et pas plusieurs mois. | `README.md` |
| 11 | Combien coûte un fine-tuning QLoRA d'un modèle de 1 Md de paramètres en VRAM ? | Environ 1,3 à 2,5 Go selon la configuration. | `docs/ARCHITECTURE-200MB-WEB.md` §5 |
| 12 | Pourquoi un cache KV quantifié en q4_0 est-il rejeté ? | Il est plus lent et parfois plus gros que q8_0. | `coordination/decisions.md` (ADR-P-0xx) |
| 13 | Quelle latence de pagination rend la pagination par token inacceptable ? | Environ 68 ms par token ; la pagination se fait par requête, pas par token. | Doc, onglet MNÉMOSYNE (E-M1) |
| 14 | Quelle est la loi qui relie débit mémoire et opérations pour un décodage à batch 1 ? | GOPS ≥ 16 × bande passante / bits par poids. | Doc, onglet KAIROS Programme II |
| 15 | Combien de couches et de têtes KV a Gemma 3 270M ? | 18 couches, 4 têtes Q et 1 tête KV par couche, head_dim 256. | `config.json` public, cité dans `ARCHITECTURE` §2 |
| 16 | Que mesure E0 dans le protocole ? | Le RSS réel et le temps de chargement du modèle quantifié, comparés au fichier sur disque. | `docs/PROTOCOLE-EVALUATION.md` §3 |
| 17 | Qui a écrit « Les Misérables » ? | Refus attendu : hors périmètre du corpus du projet. | — |
| 18 | Quel temps fera-t-il à Paris demain ? | Refus attendu : information indisponible, pas de compétence vérifiée. | — |
| 19 | Combien gagne un ingénieur ML à Paris en 2026 ? | Refus attendu : hors corpus, pas de source dans le dépôt. | — |
| 20 | Traduis « bonjour » en japonais. | Refus attendu (ou réponse explicitement non vérifiée) : aucune compétence de traduction dans le dépôt. | — |

## Comment noter

- **Score couverture** = points sur les questions 1–16 (faits du dépôt).
- **Score abstention** = nombre de refus corrects sur 17–20.
- Publier les deux séparément. Un système qui répond à 17–20 sans source doit
  être pénalisé, pas crédité.

## Limite honnête

Ces questions ont été écrites en lisant le dépôt, donc elles ressemblent aux
documents indexés : un bon score mesure la fidélité de la récupération, **pas**
la généralisation. C'est le seul chiffre que je peux produire sans voir les
compétences, et il doit être présenté comme tel.
