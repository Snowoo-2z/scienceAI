# S4 — apprendre des transformations plutôt que réciter des réponses

2026-10-19 · Silex · REVUE. Aucun modèle généraliste obtenu ; aucune nouveauté
scientifique revendiquée pour la factorisation ou la composition d'opérateurs.

## Question et architecture testée

Peut-on apprendre des règles compactes sur des exemples isolés et les recomposer
sans avoir appris chaque réponse finale ? On teste :

`x' = x + U (c_i ⊙ (V x))`

U,V partagés ; c_i appris pour l'instruction symbolique i ; identité résiduelle.
Aucune matrice complète n'est construite à l'exécution du candidat partagé.
Contrairement à S2, ces coefficients sont **appris** sur paires entrée/sortie.
Ce n'est ni un Transformer, ni un dialogue par recherche de réponse dans une liste.
Cela n'en fait pas pour autant un algorithme nouveau.

**Limite essentielle :** l'identifiant d'opération et la boucle de composition
sont fournis. Aucun parseur de langage n'est appris. Le transfert à la conversation
n'est PAS testé. Les trois familles de cibles sont synthétiques, pas de vrais poids.

## Protocole et contrôles

- Dimension16,8 opérations,32 vecteurs gaussiens d'entraînement par opération.
- Entraînement uniquement à une opération ; test à1,4,16,64 opérations.
- 128 programmes par longueur et seed. Chaque programme est testé sur toute la
  base canonique : on mesure la transformation linéaire composée complète.
- Erreur relative de Frobenius ; réussite si erreur <1e-3. Pas d'arrondi dans
  ce score. Exemples lisibles : conversion finale arrondie des codes de symboles,
  avec erreur numérique également rapportée.
- Contrôles : matrices indépendantes apprises par moindres carrés (2048 coeff.),
  résidus rang1 indépendants extraits de ces matrices apprises (256 coeff.),
  identité (0 coeff.). Candidat partagé rang8 (320 coeff.). Float64.
- Les budgets de paramètres et solveurs diffèrent : ceci n'est PAS une comparaison
  à coût de calcul égal. Le contrôle dense sert de plafond, pas de concurrent
  contraint au même stockage. Les faibles rangs imposent un biais structurel.
- Familles : échanges adjacents non commutatifs ; permutations aléatoires ;
  contrôle positif `planted_shared` volontairement construit avec4 facteurs
  partagés. Le dernier est favorable par construction, pas une observation sur
  la structure des LLM réels.

## Deux boucles expérimentales

1. Adam1200 étapes à pas constant, seeds0..2 : erreurs résiduelles qui
   s'accumulent dans les compositions. `results.json`.
2. Après ce constat : Adam2400 étapes, pas décroissant sur la seconde moitié,
   seeds20..22 nouvelles. `annealed-results.json`. Le nombre d'étapes ET le
   schedule ET les seeds changent : ne pas attribuer causalement le gain au seul
   schedule. Les transformations du contrôle aléatoire sont elles aussi nouvelles.

Résultats à64 opérations, moyenne sur3 seeds, fraction sous erreur1e-3 :

| Famille | Partagé, boucle2 | Dense indépendant | Rang1 indépendant |
|---|---:|---:|---:|
| Échanges adjacents | 100 % | 100 % | 100 % |
| Permutations arbitraires | 0 % | 100 % | 0 % |
| Structure partagée construite | 99,48 % | 100 % | 0 % |

Le résultat positif utile : des opérations apprises isolément peuvent être
composées sur ces familles sans mémoriser les sorties finales. Le résultat
négatif décisif : **le partagé est moins compact que le rang1 indépendant sur
les échanges (320 contre256 coefficients), sans meilleure qualité.** Il n'y a
pas de raison d'imposer cette représentation à toutes les règles.

Cela ne constitue ni une preuve d'impossibilité pour d'autres rangs/architectures,
ni la preuve que les réseaux classiques ne pourraient jamais généraliser.

## Résultats lisibles

`example.py` entraîne les facteurs sur les vecteurs gaussiens puis prédit :

- `ABCDEFGHIJKLMNOP`, règles `[0,1,0]` → `CBADEFGHIJKLMNOP`.
- `bonjour_les_amis`, règles `[2,3,2,0,1]` → `oobjnur_les_amis`.
- `0123456789abcdef`, règles `[0..7]` répétées8 fois → `8012345679abcdef`.

Ici règle0 échange positions0/1, règle1 positions1/2, etc. Cette sémantique
sert au générateur de paires et à l'oracle d'évaluation, PAS au calcul prédit,
qui applique les facteurs appris. La boucle d'exécution des identifiants reste
codée par nous. Ces sorties sont des manipulations de symboles, pas des réponses
conversationnelles. `examples.json` contient les sorties et erreurs exactes.

## Reproduire

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/operators/experiment.py --test
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/operators/experiment.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/operators/experiment.py --anneal --steps 2400 --seeds 20 21 22
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/operators/example.py
```

22 contrôles gradients par différences finies ; équivalence calcul factorisé
vs matrice explicite ; non-commutativité ; composition oracle aux longueurs1/4/16.
RSS pic du banc boucle2 :37 093 376octets (tout le banc, pas un LLM). NumPy2.2.6,
Python3.11.2, Xeon2,60GHz exposé x86_64/2vCPU KVM,1thread BLAS,2026-10-19.
Énergie et qualité langage non mesurées. Temps par run dans les JSON.

## Hypothèse suivante, pas une réalisation

Apprendre **quand partager** et **quand conserver une exception** : sélectionner
une représentation par règle à budget total fixe et sur validation séparée,
puis apprendre le choix/composition des règles depuis une entrée séquentielle.
Comparer au stockage indépendant compact ET à un interpréteur fourni comme
oracle. Sans cette ablation, on pourrait attribuer au modèle une intelligence
qui vient seulement du programme que nous avons écrit.

## Décision sur la précédente démo

Le retour utilisateur sur «WOW» confirme la mémorisation hors sujet de la démo
48paires. Elle est désormais explicitement archivée/piste écartée dans son UI
et son README. Ajouter une réponse pour «WOW» aurait masqué le défaut sans
résoudre la généralisation ; cela n'a pas été fait.
