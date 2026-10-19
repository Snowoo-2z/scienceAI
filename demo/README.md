# Démo — l'assistant qui répond sans modèle de langue

**Ce que c'est.** Une petite application web qui répond à des questions sur le
projet, avec une source et un extrait de preuve. **Aucun modèle de langue,
aucun réseau, aucun GPU.** C'est le « mode compétences » d'ALEXANDRIE réduit à
sa plus simple expression : 12 compétences vérifiées + un moteur de recherche
sur 1 880 passages.

**Comment l'essayer.** L'aperçu est déjà lancé (port 8000). Sinon :

```bash
python3 demo/server.py --port 8000
# ouvrir http://localhost:8000
```

**Exemples de réponses** (recopiées telles quelles) :

> **Q.** Pourquoi Gemma 3 270M est-il hors budget ?
> **R.** Gemma 3 270M pèse 253 Mo en Q4_K_M et consomme 286–296 Mo de RSS,
> au-dessus du budget de 200 Mo : c'est le vocabulaire de 262 144 tokens qui
> domine.
> *source : journal.md · preuve : « Gemma 3 270M : 253 Mo de fichier, 286–296
> Mo de RSS, hors »*

> **Q.** Combien de RAM pour un cœur de 60 M et 10 Md de paramètres ?
> **R.** Un cœur de 60 M + une bibliothèque de 10 Md de paramètres tient en
> 69 Mo (2,5 Go de disque), à ~161 tok/s et ~19 J par réponse de 200 tokens.
> *source : README.md*

> **Q.** Écris-moi un poème sur la pluie.
> **R.** Je n'ai pas de compétence vérifiée pour cette question. Voici le
> passage le plus proche du corpus, sans garantie.
> *source : README.md · score 0.0*

Le troisième cas est le plus important : **il refuse au lieu d'inventer.**
C'est la propriété qu'on cherche et qu'un modèle de langue ne garantit pas.

## Ce que ça mesure

| Mesure | Valeur | Conditions |
|---|---:|---|
| passages indexés | 1 880 | 10 documents du dépôt |
| construction de l'index | 34 ms | Python 3.11, 1 cœur |
| temps par question | 13 ms | 1 cœur, sans cache |
| jeu de 20 questions, 1er passage | 16/20 | jeu écrit par moi, sans réglage |
| jeu de 20 questions, après 4 corrections | 20/20 | **jeu vu** → ne prouve rien |
| refus sur 8 questions hors corpus | 8/8 | trivial : aucune skill ne matche |

**Avertissement honnête.** Les 20/20 ne sont pas une mesure de généralisation :
j'ai écrit les questions, puis ajouté 4 compétences après avoir vu les échecs.
Le chiffre qui compte est 16/20 avant réglage, et même celui-là est optimiste
puisque le jeu est de moi. Il faut un jeu écrit par un autre agent ou par
l'humain pour avoir un chiffre défendable. C'est une demande de l'onglet 14.

## Ce que ça ne fait pas

- pas de conversation : chaque question est indépendante ;
- pas de génération : les réponses sont écrites à la main dans les skills ;
- pas de couverture ouverte : les 8 questions hors corpus sont refusées
  (poème, photosynthèse, traduction, météo, code, opinion, résumé) ;
- pas de mémoire : l'assistant n'apprend rien entre deux requêtes.

C'est exactement la frontière que le document décrit : **le vérifiable se
stocke en compétences, le reste demande un modèle de langue.** La démo rend
visible la partie gratuite — et son plafond.

## Fichiers

| Fichier | Rôle |
|---|---|
| `demo/server.py` | serveur HTTP stdlib, route `/ask` |
| `demo/index.html` | interface de chat (aucune dépendance) |
| `tools/demo_assistant.py` | moteur : retrieval TF-IDF + 12 skills vérifiées |
| `tools/eval_common.py` | évalue le moteur sur `demo/eval-questions.json` |
| `demo/eval-questions.json` | 20 questions, 12 dans le corpus, 8 ouvertes |
| `results/eval-common.txt` | dernier résultat d'évaluation |
| `results/demo-assistant.txt` | 10 questions/réponses types |
| `results/demo-probe.txt` | sonde de couverture |
