# ARCHIVE — Silex : démonstration de mémorisation, piste écartée

**Décision après retour utilisateur : ne pas développer ce réseau comme voie vers le généraliste.** Les artefacts restent pour reproductibilité.

2026-10-19. Démonstrateur de laboratoire destiné à rendre les résultats visibles
sans savoir programmer. **Ce n'est pas le généraliste recherché.**

## Essayer

```sh
# Depuis la racine du dépôt, après installation de experiments/silex/requirements.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/dialogue/server.py
# Ouvrir le port 8080 dans l'aperçu Arena, ou http://localhost:8080 sur sa propre machine.
```

Le serveur lit uniquement `weights.npz` et `metadata.json` pour générer. Il ne
cherche pas une réponse dans `corpus.json`. Le navigateur peut charger
`responses.json` pour afficher des exemples enregistrés, mais chaque envoi de
question déclenche une vraie inférence. Aucun appel réseau à un modèle externe,
aucune réécriture de la réponse par un grand modèle. Le décodage est déterministe.
Les échanges précédents ne sont pas envoyés au modèle. L'application ne persiste
pas les questions. Serveur local de démonstration, pas service public sécurisé.

## Ce qui a été entraîné

- 48 paires pédagogiques synthétiques, rédigées pour cette expérience, provenance
  déclarée dans corpus.json. 24 petits thèmes, pas un corpus généraliste.
- Vocabulaire construit UNIQUEMENT depuis l'entraînement. Les mots inconnus
  deviennent `<unk>`, et sont indiqués à l'utilisateur.
- Encodeur de question en sac de mots ; décodeur récurrent `tanh` partagé,
  état de dimension 64. Le contexte de question est réinjecté à chaque pas.
- Entraînement supervisé de bout en bout par rétropropagation à travers le temps,
  teacher forcing, Adam, 800 étapes full batch, seed42, float32, CPU NumPy.
- 38 514 paramètres stockés, 154 056 octets de tableaux. Ce n'est PAS un plafond
  décidé pour le projet : juste une expérience rapidement testable.
- Ni attention ni Transformer. Ce RNN de référence est un mécanisme connu,
  **pas une architecture nouvelle revendiquée**. Les opérateurs S2, la mémoire
  plastique S3 et la profondeur adaptative ne sont PAS intégrés ici.

## Reproduire et vérifier

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/dialogue/test_model.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/dialogue/model.py train --epochs 800
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/dialogue/model.py eval
```

Entraîner remplace le petit checkpoint local. Le checkpoint livré (152 Kio
avec conteneur NPZ) est versionné volontairement pour que le démonstrateur soit
essayable sans réentraînement. Pas de téléchargement de poids externes.

`metadata.json` contient empreintes SHA-256 corpus/checkpoint, taille, temps et
courbe de perte. `responses.json` contient TOUTES les réponses des 48 exemples,
des 6 paraphrases réservées et de 6 nouvelles questions, sans sélection des
seules réussites. Le rendu espace/apostrophe est une simple détokénisation.

30 contrôles par différences finies vérifient les gradients (erreur max lors du
run ~2,1e-11 en float64). Tests supplémentaires : mots inconnus, encodage vide,
génération bornée/déterministe, limite d'ordre du sac de mots. HTTP testé avec
host de preview, cas valides, entrées invalides et 404 ; JS vérifié avec node.
Pas de test visuel automatisé par navigateur effectué.

## Résultats bruts

| Question | Sortie réelle |
|---|---|
| bonjour | bonjour ! comment puis - je aider ? |
| explique-moi la gravité | la gravité est une interaction attractive entre les masses. |
| peux-tu raconter une histoire ? | non. ce prototype fonctionne sans accès à internet. |
| quelle est la capitale du japon ? | la capitale de l'italie est rome. |
| combien font deux plus trois ? | deux plus deux font quatre. |

48/48 cibles reproduites sur entraînement, 4/6 sur paraphrases réservées.
**Mémorisation réussie, généralisation très insuffisante.** Les six nouvelles
questions mettent en évidence des échecs ; ce n'est pas un benchmark généraliste.
Le choix de 800 étapes précède la lecture des scores de paraphrases. Une seule
seed, pas de conclusion robuste entre architectures. L'interface montre deux
réussites et deux erreurs et permet de poser librement d'autres questions.

Mesures Linux x86_64, Xeon 2,60 GHz exposé/2vCPU KVM, Python3.11.2,
NumPy2.2.6, un thread BLAS, 2026-10-19 : entraînement ~3,56 s / RSS pic38,23 Mo ;
évaluation dans un processus séparé RSS pic32,78 Mo. Serveur après requêtes
~36,23 Mo de RSS courant lors du test. Ces nombres décrivent ce **jouet**, pas
un modèle généraliste ni tous les matériels. OS, navigateur et mémoire cgroup
non mesurés ici. Énergie non mesurée.

## Limite structurelle identifiée

Le sac de mots efface l'ordre : « la france est la capitale de paris » et
« paris est la capitale de la france » donnent le même vecteur. La contradiction
ne peut donc pas être résolue par ce seul encodeur, même avec plus d'entraînement.
Ce contrôle volontairement simple ne doit pas être adopté comme produit.

## Prochaine étape et décision de données

1. Remplacer le sac de mots par un encodeur séquentiel et comparer sur paires
   dont l'ordre change le sens, en conservant des questions réservées.
2. Tester l'intégration d'une mémoire plastique/cache sous budget total fixe.
3. Utiliser des conversations variées avec provenance/licence et séparation
   train/test par conversation ; le corpus de 48 exemples ne suffit pas.

Source identifiée, **aucune donnée externe utilisée pour ce checkpoint** :
`bofenghuang/vigogne`, commit `67ea8a2a06f1ddd5652da274c9d537e2223468bc`.
Le fichier DATA_LICENSE annonce CC BY-NC 4.0 ; docs/data.md décrit notamment
Alpaca traduit en français et des extraits OASST. Les droits spécifiques des
sources et traductions doivent aussi être vérifiés. Ne pas importer cette
source dans un futur produit commercial par défaut.

- https://github.com/bofenghuang/vigogne/blob/67ea8a2a06f1ddd5652da274c9d537e2223468bc/DATA_LICENSE
- https://github.com/bofenghuang/vigogne/blob/67ea8a2a06f1ddd5652da274c9d537e2223468bc/docs/data.md

Décision demandée au porteur : autoriser une piste de données explicitement non
commerciales pour la recherche, ou exiger une piste compatible commercial.
Cela ne suffit pas à prouver qu'un corpus résoudra la qualité du modèle.
