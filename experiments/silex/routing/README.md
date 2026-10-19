# S6 — Sans oracle, la compréhension devient le point de rupture

**Résultat négatif, pas un assistant généraliste.** Une bibliothèque compacte
exécute correctement des opérations lorsqu'on lui fournit leur identifiant.
Cela ne signifie pas qu'elle comprend les demandes. Deux expériences ici
retirent cet oracle, puis testent une décomposition intention/arguments.
Aucune réponse conversationnelle pré-écrite, aucune revendication de nouveauté.

## Protocole initial fixé avant les résultats

- Français contrôlé, huit échanges adjacents sur une liste de 16 éléments.
  Les frontières des étapes sont **fournies**, pas apprises.
- `data.py` : 120 exemples train (80 supportés/40 non supportés), 68 validation
  (32/36), 136 test (64/72). Chiffres et mots, patrons de phrases distincts.
  Les requêtes hors domaine incluent négations, suppressions, additions et
  questions générales. Même corpus écrit par nous pour les trois seeds :
  **ce ne sont pas trois évaluations linguistiques indépendantes**.
- 512 caractéristiques signées BLAKE2b sur n-grammes de caractères 2–4,
  normalisation NFKC/casefold et L2 ; régression softmax float32, Adam 500 pas,
  lr .03, régularisation 1e-4 ; seeds 7, 17, 27. Deux ablations annoncées :
  huit classes sans négatifs, neuf classes avec classe de refus.
- Seuil : couverture maximale parmi les scores validation sous une limite
  d'erreur **empirique** de 5 %, avec au moins 16 acceptations. À défaut, rejet
  total. Ce n'est ni une garantie statistique ni une sûreté hors distribution.
- Opérateurs appris par moindres carrés sur 32 paires gaussiennes/opérateur,
  validation indépendante de 48 vecteurs, compression S5 `independent`.
  Exécution via SQLite/Paged, sans fallback vers le label vrai.
- 128 programmes par longueur 1/4/16/64 et seed, instructions supportées du
  test, mêmes programmes entre politiques oracle/top1/selective. Le résultat
  est vérifié sur toute la matrice identité 16×16, erreur relative <1e-4.
  Les demandes non supportées sont évaluées séparément, **pas mélangées dans
  ces programmes**. Des erreurs de routage peuvent se compenser : les erreurs
  d'étapes et la justesse finale sont donc toutes deux enregistrées.

## Résultats initiaux (`results.json`)

Les scores de requêtes suivants sont identiques pour les trois seeds :

| Routeur / décision | Supportés correctement acceptés | Hors domaine acceptés à tort | Justesse parmi acceptés |
|---|---:|---:|---:|
| 8 classes / top1 | 76,56 % | 100 % | 36,03 % |
| 8 classes / seuil | 0 % | 0 % | aucune acceptation |
| 9 classes / top1 | 59,38 % | 69,44 % | 40,43 % |
| 9 classes / seuil | 7,81 % | 55,56 % | 11,11 % |

Avec neuf classes, la calibration accepte 25 validations dont une fausse
(4 % d'erreur), mais 45 tests dont 40 faux (**88,89 % d'erreur**).
La confiance softmax est trompeuse après changement de formulation.
Sans rejet, les programmes de 64 étapes ont **0/384 succès**, contre
384/384 avec oracle. Les politiques à rejet n'en réussissent aucun non plus.
Ce résultat condamne cette solution comme contrôleur fiable, pas toutes les
architectures à mémoire externe.

## Ablation exploratoire après observation du premier échec

`typed.py` sépare intention et arguments : classifieur binaire appris après
masquage des nombres, mais **reconnaissance des nombres et liaison des deux
DERNIERS arguments écrites à la main**. La condition d'adjacence/domaine est
également fournie. Ce n'est pas un parseur appris de bout en bout.
Même train/validation, mêmes hyperparamètres ; 4 104 octets de paramètres.
Le test initial est désormais **diagnostique**, car cette architecture a été
choisie après ses résultats. Un challenge supplémentaire de 32 positifs et
54 négatifs est écrit avant le premier lancement de cette ablation. Il reste
interne au même auteur, pas aveugle ni une preuve générale.

| Test de l'ablation | Supportés correctement acceptés | Hors domaine acceptés à tort | Justesse parmi acceptés |
|---|---:|---:|---:|
| Test initial revisité | 100 % | 66,67 % | 57,14 % |
| Nouveau challenge | 50 % | 37,04 % | 44,44 % |

Ici top1 et seuil donnent les mêmes décisions. Validation : 33 acceptations,
une fausse (3,03 %). La structure résout les anciens arguments, et tous les
anciens programmes supportés passent, mais **pas la sémantique de refus**.
Sur le nouveau challenge, les programmes de 64 étapes échouent tous.

Exemples réellement routés, seed 7, politique sélective :
- « garde les éléments 1 et 2 à leur place » → échange 1 et 2,
  confiance 0,9659 : **contraire à la demande** ;
- « explique comment échanger 1 et 2, mais sans rien modifier » → échange 1
  et 2, confiance 0,9449 : **exécute au lieu d'expliquer**.

Les négations peuvent donc échouer même quand la sortie a le bon type,
les arguments sont valides et le moteur d'exécution est exact. Une sortie
structurellement valide ne constitue pas une preuve de compréhension.

## Mémoire, coût et limites

Budget total **payload numérique** de 65 536 octets : poids+biais routeur
(16 416 / 18 468 / 4 104 octets), puis cache opérateurs sur le reste. L'oracle
ne paie pas de routeur ; ne pas interpréter sa latence comme un concurrent
réaliste. SQLite possède aussi son cache S5 de 256 Kio ; code, objets Python,
chaînes, activations, matrices, stockage disque et cache noyau ne sont pas
comptés dans le payload. Les fonctions/types fournis ne sont pas « gratuits ».

Pics RSS processus **entraînement + évaluation combinés** : 39 927 808 octets
(initial) et 39 325 696 octets (ablation), pas des mesures d'un service déployé.
Python 3.11.2, NumPy 2.2.6, float32 routeur/float64 opérateurs, CPU Xeon x86_64,
OpenBLAS à un thread, 2026-10-19. Les temps incluent le calcul de caractéristiques
à chaque instruction mais pas la construction de la bibliothèque. Une seule
exécution par seed ; aucune mesure d'énergie, aucun plafond cgroup 200 Mo imposé,
aucune conclusion de portabilité à tout matériel ou de qualité généraliste.
Les bibliothèques temporaires sont supprimées après chaque run.

## Reproduire

Depuis la racine, avec NumPy 2.2.6 installé dans `.venv` :

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/routing/experiment.py > /tmp/s6.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/routing/typed.py > /tmp/s6-typed.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s experiments/silex/routing -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s experiments/silex/paged -p 'test_*.py' -v
```

Six tests routage : disjonction des splits/challenge, caractéristiques Unicode,
gradients, calibration/rejet, limites de la liaison symbolique, absence de
fallback oracle et respect du budget. Sept tests S5 passent aussi.

## Conséquence pour l'équipe

Ne pas assembler une démo généraliste sur ce routeur et ne pas annoncer une
sûreté sur la seule validation. Garder Paged comme composant de stockage testé.
Le prochain contrôleur doit distinguer action, mention, négation et conditions,
pas seulement choisir un ID. Demander une évaluation aveugle à un autre agent,
avec tests minimaux contrastifs et programmes mixtes (supportés/non supportés),
avant de conclure sur une nouvelle architecture. Ajouter des règles pour ces
phrases test précises serait masquer l'échec, pas résoudre le langage.
