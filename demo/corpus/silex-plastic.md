# S3a–c : mémoire associative, corrections et interférence

2026-10-19 · Silex · **REVUE**. Micro-expérience d'adaptation en ligne,
pas entraînement de langage, pas invention d'une nouvelle règle d'apprentissage.

## Pourquoi cette expérience

S2 montrait seulement qu'on peut composer des opérateurs avec une mémoire fixe.
S3a–c teste une capacité : mémoriser des associations, en corriger une moitié,
puis rappeler les valeurs corrigées ET celles qui devaient rester inchangées.
Cela explore une brique possible de mémoire récurrente/plastique. L'ambition
reste généraliste sous 100 Mo ; aucun nombre maximal de paramètres n'est fixé.

## Protocole

- Clés gaussiennes normalisées, dimension 64 ; valeurs aléatoires parmi 16 classes.
- 16, 64, 128 ou 256 associations écrites séquentiellement une fois.
- Correction d'une moitié choisie aléatoirement, vers une classe différente.
- Requête : clé d'origine, ou ajout de bruit gaussien sigma=0,1 par coordonnée
  puis renormalisation. Ce bruit n'est pas une paraphrase en langage naturel.
- Hebb : `M += outer(k,v)` ; delta : `M += beta * outer(k, v-k@M)`.
  Clés unitaires, état float32 de 64×16 = 4096 octets.
- FIFO float32 : 12 clés et valeurs one-hot = 3840 octets.
- FIFO-q8 : clés normalisées arrondies après multiplication par 127, valeurs
  uint8, 63 paires = 4095 octets. Mise à jour si clé quantifiée déjà présente,
  sinon éviction circulaire. Recherche par similarité cosinus. Ce contrôle est
  plus compétitif que stocker les valeurs one-hot ; aucune garantie d'absence
  de collision de quantification. Curseur/objets/temporaire exclus du payload.
- Tous reçoivent les mêmes données pour un couple seed/scénario. L'évaluation
  compare les labels, n'actualise pas la mémoire et ne retourne pas les réponses
  au système. Les données/oracles sont conservés par le banc de test, pas par
  la mémoire testée. Aucune tâche d'abstention sur clé inconnue n'est testée.
- Seeds exploratoires 0..4, beta = 1 / 0,5 / 0,25. Sélection de beta0,25 sur
  l'exactitude globale du scénario 128 associations/bruit0,1, puis fixation
  annoncée dans le Doc AVANT les seeds de confirmation 10..14.
- La confirmation porte sur de nouveaux tirages de la même distribution,
  PAS sur une généralisation hors distribution ou une capacité de langage.

## Configuration et reproduction

Xeon @2,60 GHz exposé, 2 vCPU KVM, x86_64, Linux6.1.158+, Python3.11.2,
NumPy2.2.6. OPENBLAS_NUM_THREADS=1. Aucun GPU, aucun poids externe.
Les timings sont ceux du banc (écriture + rappel avant/après corrections), pas
un débit token/s. Pas de capteur énergétique. RSS par getrusage : environ 35 Mo
pour le processus du banc entier, **pas un LLM complet** ni une mesure cgroup.

```sh
python -m venv .venv
.venv/bin/pip install -r experiments/silex/requirements.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/plastic_memory.py --self-test
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/plastic_memory.py
# Exploration, répéter avec beta=0.5 et 1.0
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/plastic_memory.py --noise 0.1 --beta 0.25
# Confirmation, répéter avec beta=1.0
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/plastic_memory.py --noise 0.1 --beta 0.25 --seed-start 10
```

Les six JSON contiennent 480 exécutions de méthodes/scénarios/seeds au total,
mais certains contrôles sont répétés à l'identique dans les balayages beta :
**ce ne sont pas 480 observations statistiquement indépendantes**. Écarts-types
calculés sur cinq seeds, pas des intervalles de confiance. Pas de comparaison
à calcul strictement égal : budgets de tableaux proches, opérations différentes.

## Résultats et décisions

### Cycle 1 : la correction fonctionne, l'oubli aussi

Clés propres, beta1, seeds0..4 :

| Associations | Méthode | Rappel corrigées | Rappel inchangées |
|---:|---|---:|---:|
| 16 | Hebb | 40,0 % | 100 % |
| 16 | Delta | 100 % | 100 % |
| 128 | Hebb | 44,7 % | 70,3 % |
| 128 | Delta | 80,9 % | 23,1 % |

Donc pas de capacité illimitée avec un état fixe : interférence importante.

### Cycle 2 : contrôle plus fort et écriture modérée

À 64 associations/bruit0,1, FIFO-q8 fait 98,4 % global contre 73,1 % pour
delta beta1. Ne pas vendre la mémoire plastique comme meilleure que toute table.
À 128 associations/bruit0,1, beta0,25 monte à 60,3 % contre 41,9 % pour beta1
(exploration). Le cache FIFO-q8 fait 51,7 %. Ce constat motive la confirmation.

### Cycle 3 : nouvelles seeds, réglage fixé à l'avance

128 associations/bruit0,1, seeds10..14 :

| Méthode | Global moyen ± écart-type seeds | Corrigées | Inchangées |
|---|---:|---:|---:|
| Hebb | 44,84 ± 2,57 % | 38,75 % | 50,94 % |
| Delta beta1 | 43,13 ± 3,05 % | 68,13 % | 18,13 % |
| Delta beta0,25 | 55,31 ± 2,17 % | 67,19 % | 43,44 % |
| FIFO float32 | 14,53 ± 2,80 % | 24,38 % | 4,69 % |
| FIFO-q8 | 52,50 ± 2,02 % | 83,44 % | 21,56 % |

Réduire beta améliore ici la conservation sans beaucoup réduire le rappel des
corrections. L'écart de 2,81 points avec FIFO-q8 ne justifie pas une proclamation
de supériorité générale ; pas de test de significativité préspécifié.

## Ce que cela change dans la recherche

- Rejeter l'idée que profondeur virtuelle ou mémoire plastique donne de la
  capacité gratuite. Tester capacité de rappel et oubli, pas juste RAM.
- Comparer systématiquement aux caches quantifiés : contrôle simple mais fort.
- Prochaine hypothèse proposée : routeur d'écriture appris ou allocation entre
  cache compact et état plastique, avec un budget TOTAL fixe (pas 4 Ko + 4 Ko
  contre 4 Ko). Une ablation et des seeds réservées sont nécessaires.
- La combinaison avec les opérateurs S2, l'apprentissage d'un contrôleur,
  la profondeur adaptative et les évaluations de langage restent **non réalisés**.
  Il n'existe pas encore de résultat démontrant l'objectif généraliste.

## Vérifications

Self-test : rappel de clés orthogonales, remplacement exact delta, budget FIFO,
remplacement sans avancer le curseur, éviction, reproductibilité des scores,
payload FIFO-q8 et lecture/remplacement q8. S2 garde ses neuf tests d'équivalence.
