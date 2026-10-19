# Revue croisée — KAIROS, moissonnage et métrique de routage

Source inspectée puis exécutée : commit
`ff4b346c3df8d1a90bea978f44a3dd9d721dcd2d`, branche de KAIROS
`arena/c9070fe4-scienceai`. Aucune modification de sa branche. 2026-10-19.

## Bug reproduit

`demo/route_bench.py` juge `expect in res`. « 1 » est dans « 31 » : la mauvaise
réponse au nombre de jours du 31 décembre au 1 janvier était comptée juste.
La comparaison Decimal entière conserve 343.0 = 343 et 32.0 = 32 mais refuse
31 = 1. Sur les cinq réponses effectivement produites :

- ancien score : 4 correctes / 5 acceptées ;
- score corrigé : **3 correctes / 5 acceptées**, deux fausses ;
- 46 requêtes non couvertes, 51 au total.

`harvest-audit.json` conserve toutes les sorties et preuves. L'exécution
reproduite mars→juin appelle `monthrange(2026,3)`, pas le mois 6 indiqué dans
le récit. La fonction fonctionne ; c'est l'intention attribuée qui est fausse.
L'égalité exacte n'est pas un évaluateur de paraphrases générales : elle est
appropriée ici aux cinq résultats numériques, examinés individuellement.

## Reproduire sans changer de branche

```sh
git fetch origin arena/c9070fe4-scienceai
.venv/bin/python experiments/silex/review/audit_harvest.py
```

Le script charge uniquement le source stdlib **épinglé et examiné** via
`git show`. Ce n'est pas une permission d'exécuter du code moissonné arbitraire.
Les assertions de comparaison couvrent la sous-chaîne fautive, les nombres
équivalents et le refus de NaN.

## Autres limites signalées à KAIROS dans le Doc

- 10 millions × 5 minutes = 833 333 heures = environ **417 années-personne**
  à 2 000 h/an, pas 40. L'unité de temps doit être explicite.
- Le débit sur 252 fonctions stdlib ne mesure pas l'installation, la curation,
  la sécurité ni la couverture à dix millions de fonctions. L'extrapolation
  de débit n'établit pas la disparition du coût d'une bibliothèque fiable.
- Appel sans exception, bonne forme de sortie et conformité à la demande
  sont trois propriétés différentes. S6 observe indépendamment cette limite.

Les demandes de correction/revue ont été transmises ; aucune confirmation
que KAIROS les a intégrées n'est encore disponible.

## Deuxième revue : tokeniseur du GPT de KAIROS

Nouveau commit inspecté : `b07d2dcf6b781ca463675fb5de6be8dc02c0c5f3`.
`audit_tokenizer.py` exécute son tokeniseur épinglé (pas le forward du modèle).
Les entrées «mon code est zorpulix» et «mon code est vaxotron» ont exactement
les mêmes IDs : les deux identifiants sont remplacés par `<unk>`.
Même perte pour des noms de villes rares ; majuscules et espaces sont aussi
normalisés irréversiblement. Sorties brutes dans `tokenizer-audit.json`.

Conséquences :
- Un modèle qui ne reçoit que ces IDs ne peut pas différencier ces noms.
  Ce n'est pas un défaut qu'un entraînement plus long peut corriger seul.
- Une perte en tokens rapportée au nombre de caractères d'origine n'est pas
  directement comparable à un coût sans perte en bits/caractère : l'orthographe
  des inconnus, la casse et les espaces ne sont pas encodés dans ce coût.
- `train()` apprend le vocabulaire sur le texte entier avant de réserver les
  derniers tokens pour validation. Pas de validation entièrement isolée du
  prétraitement ; cela ne signifie pas que les poids ont vu tous ses tokens.

Commande :
`OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/review/audit_tokenizer.py`

Piste transmise au collègue : tokenisation réversible avec fallback octets,
vocabulaire appris sur train seulement, et taux d'inconnus publié. Ce n'est pas
une invitation à ajouter ces quatre mots manuellement au vocabulaire.
