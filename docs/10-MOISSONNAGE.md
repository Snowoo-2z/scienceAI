# 10 — « Il faudrait écrire chaque compétence à la main » : la critique est juste

**2026-10-19 · KAIROS · branche `arena/c9070fe4-scienceai`**
Code : `demo/harvest.py`, `demo/route_bench.py`
Résultats : `results/harvest-index.txt`, `results/harvest-routing.txt`

---

## 1. Le calcul que je n'avais pas fait

Le Programme II proposait une bibliothèque de ~10 M de compétences vérifiées.
Je n'avais jamais chiffré leur **écriture**. Le calcul :

| Hypothèse | Temps total | En années-personne |
|---|---:|---:|
| 5 minutes par compétence | 833 000 h | **40 ans** |
| 1 minute par compétence | 167 000 h | 8 ans |
| 10 secondes par compétence | 27 800 h | 1,3 an |

Même à 10 secondes — impossible pour une procédure testée et documentée — on
n'y arrive pas. **L'idée « écrire 10 M de compétences » est morte.** Elle
l'est d'autant plus que le coût de curation (dédoublonnage, tests,
provenance) est plus lourd que l'écriture elle-même.

C'est écrit noir sur blanc dans `docs/06` §7 : « le coût de construction de la
bibliothèque est probablement le poste le plus lourd ». Je l'avais identifié
et je ne l'avais pas traité. La critique a raison.

## 2. Ce qui remplace l'écriture : le moissonnage

Le monde contient déjà des millions de fonctions que leurs auteurs ont
écrites, testées et maintenues. Le travail n'est donc pas de les rédiger,
mais de les **découvrir, décrire, tester et indexer**.

`demo/harvest.py` le fait pour la bibliothèque standard : il lit les
signatures, génère une fiche (nom, signature, docstring, arité), fabrique un
ou plusieurs jeux d'arguments automatiques, appelle la fonction et ne garde
que celles qui s'exécutent.

Mesuré dans la sandbox, sur 11 modules :

| Métrique | Valeur |
|---|---:|
| Fonctions publiques vues | 252 |
| Fiches produites | 242 |
| Enveloppes validées automatiquement | **133** |
| Échecs (signature ou arguments) | 109 |
| Temps total | 32 ms |
| Débit | **7 596 fiches/s** (0,13 ms par fiche) |
| Taille d'une fiche | 249 octets |
| Projection à 10 M de fiches | **0,37 h** de moissonnage, 2,5 Go d'index |

Le taux de réussite de 55 % vient de mon générateur d'arguments, volontairement
naïf (nom du paramètre + profils types). Avec les annotations de type et les
docstrings, il monterait. C'est de l'ingénierie, pas de la recherche.

**Donc : le coût d'écriture tombe de 40 ans à quelques heures.** La
bibliothèque n'a plus besoin d'être écrite à la main. Mais ce n'est que la
moitié de l'histoire.

## 3. Le vrai goulot, mesuré : le routage

Les 133 fiches moissonnées sont passées sur le banc des 51 questions, sans
aucune compétence écrite à la main :

| Résultat | Nombre |
|---|---:|
| Justes | **4** |
| **Fausses, avec une preuve** | **1** |
| Non couvertes | 46 |
| Précision quand le routeur répond | 4/5 |

La fausse réponse est le résultat le plus important de ce document :

> « Combien de jours entre le 15 mars 2026 et le 15 juin 2026 ? »
> → **31** (la fonction appelée est `calendar.monthrange(2026, 6)`, qui
> renvoie le nombre de jours de juin).
> Preuve affichée : `monthrange(2026, 6) → 31 jours`. **C'est vérifié et
> c'est faux.**

Le vérificateur a fait son travail : la fonction a bien tourné, sans erreur.
Il n'a pas fait le travail qui compte : vérifier que **la question posée**
est celle à laquelle la fonction répond. Un test d'exécution ne juge pas
l'intention.

## 4. Ce que ça change dans le programme

L'écriture des compétences n'est plus le problème. Le problème est **la
sélection et le remplissage des arguments** — c'est-à-dire exactement le rôle
du cortex. La bonne nouvelle : c'est une tâche beaucoup plus étroite que
« stocker la connaissance du monde ».

Un routeur n'a pas besoin de savoir que Paris est en France ; il a besoin de
savoir que « combien de jours entre deux dates » appelle une soustraction de
dates, pas `monthrange`. C'est de la compréhension de **forme**, pas de
contenu.

La proposition révisée, et elle est plus modeste :

> **Un petit cortex qui ne connaît rien, mais qui sait choisir et remplir.**
> La connaissance et les compétences vivent dans la bibliothèque moissonnée
> et auto-étendue ; le cortex n'apprend qu'à router.

C'est plus faible que « un cortex de 8 B + 10 M de skills bat un modèle 10×
plus gros ». C'est aussi beaucoup plus testable, et c'est la seule version
qui survive à la critique de coût.

## 5. Combien de compétences faut-il vraiment ?

Pas 10 M. Le nombre utile est le nombre de **requêtes distinctes** que le
système rencontre réellement, pas le nombre de procédures imaginables. La loi
de Heaps donne une croissance sous-linéaire : `distincts ≈ N^β` avec
β ≈ 0,5–0,6.

| Requêtes servies | Requêtes distinctes (β = 0,5) | (β = 0,6) |
|---:|---:|---:|
| 10⁴ | 100 | 251 |
| 10⁶ | 1 000 | 3 981 |
| 10⁸ | 10 000 | 63 096 |
| 10¹⁰ | 100 000 | 1 000 000 |

Autrement dit : servir un milliard de requêtes ne demande pas un milliard de
compétences, mais de l'ordre de **10 000 à 60 000**. C'est un objectif
atteignable par auto-extension, et il est compatible avec un index de
quelques dizaines de Mo.

## 6. Le mécanisme d'auto-extension, en une phrase

> Une compétence n'est écrite **que lorsqu'une requête échoue**, une seule
> fois, puis testée, puis réutilisée.

Le coût marginal d'une nouvelle compétence est donc **une génération**, pas
cinq minutes d'humain. Et si la requête ne revient jamais, on n'a rien écrit :
la dépense suit l'usage, elle ne le précède pas.

## 7. Critères d'abandon, écrits à l'avance

Le programme révisé meurt si, sur un banc de **500 questions de routage
écrites par un autre agent** (question → fonction + arguments attendus) :

| Critère | Seuil | Conséquence |
|---|---:|---|
| Précision de sélection | < 70 % | abandonner I-1 |
| Taux de fausses réponses acceptées par le vérificateur | > 5 % | le vérificateur ne suffit pas ; abandonner |
| Croissance des requêtes distinctes | > N^0,7 | le coût de la bibliothèque explose ; abandonner |
| Coût de curation par compétence | > 0,10 € | l'auto-extension ne tient pas |

Aucun de ces seuils n'est atteint ou invalidé aujourd'hui : **ils ne sont pas
mesurés**. C'est la prochaine expérience, et elle coûte moins de 1 000 €.

## 8. Ce qui est abandonné, ce qui reste

**Abandonné :**
- « 10 M de compétences écrites à la main » ;
- construire la bibliothèque **avant** l'usage ;
- l'idée qu'un test d'exécution suffit à vérifier une réponse.

**Ce qui reste, et qui est maintenant mesurable :**
- le moissonnage du code existant (7 596 fiches/s, mesuré) ;
- l'auto-extension à l'usage (une génération par compétence nouvelle) ;
- un cortex de routage, étroit, dont la seule tâche est de choisir et remplir ;
- l'aveu quand rien ne s'applique, qui reste la seule protection contre le
  faux vérifié.
