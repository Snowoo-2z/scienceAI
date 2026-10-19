# scienceAI — branche de recherche Silex

Objectif du projet : rechercher une IA conversationnelle généraliste, avec
**100 Mo de RAM visés, jusqu'à200 Mo acceptables**, stockage disque autorisé.
**Cet objectif n'est pas atteint.** Cette branche contient des expériences
reproductibles, leurs limites et des composants exécutables, pas une promesse
qu'un petit assistant à règles serait déjà généraliste.

## Dernier prototype : apprendre des procédures par démonstrations

L'atelier S7 découvre des compositions d'opérations à partir de vos exemples,
les applique à des entrées nouvelles et demande une clarification quand ses
interprétations divergent. Il ne comprend pas une conversation libre.

```sh
python3 experiments/silex/induction/server.py --port 8082
```

Puis ouvrir le port8082 dans l'aperçu. Aucun paquet externe requis sous Linux.
Le serveur impose une limite d'espace virtuel de100Mo ; RSS mesuré~21Mo.
Ce plafond processus n'inclut pas le cache disque du système.

[Protocole, résultats, commandes et limites S7](experiments/silex/induction/README.md)

-256 procédures synthétiques testées,2048 entrées réservées ;
-procédures acquises, mais29 opérations élémentaires fournies ;
-résultats négatifs conservés : des hypothèses peuvent toutes s'accorder sur
 une réponse fausse si la bonne règle est absente du langage de recherche ;
-pas d'API distante, pas de réponses de chat pré-écrites.

## Travaux précédents et collaboration

- [S6 : échec du routage linguistique et des seuils de confiance](experiments/silex/routing/README.md).
- [S5 : représentation adaptative et bibliothèque paginée](experiments/silex/paged/README.md).
- [S4 : opérateurs appris et compositions](experiments/silex/operators/README.md).
- [Revues croisées reproductibles des métriques et du tokeniseur](experiments/silex/review/README.md).
- [Journal de coordination](coordination/journal.md).

Les autres agents travaillent sur leurs branches ; leurs modèles et serveurs
ne sont pas intégrés ici par simple affirmation. Les revues citent des commits
épinglés. L'ancienne démo48paires a été rejetée et archivée, pas renommée en
nouveau système.
