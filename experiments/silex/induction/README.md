# S7 — Acquérir une procédure, conserver l'ambiguïté, rechercher un contre-exemple

**Prototype exécuté et essayable ; pas une IA conversationnelle généraliste.**
Après l'échec du routage français S6, cette piste change de signal
 d'apprentissage : des démonstrations entrée→sortie, et non des phrases dont
on prétend comprendre l'intention. Les **29 primitives** sont écrites par nous ;
leurs **compositions sont découvertes**, pas fournies au solveur. Aucun réseau
neuronal, aucune API distante, aucune réponse conversationnelle mémorisée.

## Essai concret

```sh
python experiments/silex/induction/server.py --port 8082
```

L'atelier accepte texte, entiers et listes JSON plates. Exemple réellement testé :

- `"  BONJOUR MONDE  "` → `"bonjour monde"` ;
- `"  LUNDI  "` → `"lundi"` ;
- nouvelle entrée `"  SALUT PARIS  "` → **`"salut paris"`**.

Il reste 22 programmes compatibles. Sur d'autres entrées, ils divergent :
le système propose alors une démonstration discriminante. On peut aussi lui
fournir un contre-exemple choisi soi-même. Les programmes peuvent être
inspectés ; les résultats ne sont pas récupérés dans une table de réponses.
C'est une capacité d'apprentissage **étroite**, pas la compréhension d'une
conversation. L'ancienne démo RNN rejetée a été arrêtée.

## Mécanisme et limites exactes

1. Recherche exhaustive de compositions jusqu'à trois étapes, profondeur
   d'abord, sur une grammaire typée fermée. Vérification de chaque composition
   sur les démonstrations. Aucun label de programme fourni au solveur.
2. Conservation de **toutes les syntaxes compatibles** en SQLite, même si
   elles ont la même sortie sur les démonstrations. Les fusionner à ce stade
   supprimerait précisément les interprétations qu'il faut départager.
3. Sur une nouvelle entrée : réponse seulement si toutes les hypothèses
   restantes donnent la même sortie valide et la recherche était complète.
4. Clarification : partition des hypothèses par sortie sur des entrées
   candidates non étiquetées, choix de l'entropie maximale. C'est un critère
   sur des **syntaxes**, pas une probabilité calibrée sur les intentions.
   Les programmes sémantiquement équivalents sont surcomptés.
5. Nouvelle observation : suppression des incompatibles. Si l'ensemble
   devient vide, `revise_if_empty` peut rouvrir la recherche à profondeur4
   avec toutes les observations, sans ajouter une primitive spécifique.
6. Limites : 100000 nœuds / 10000 hypothèses ; troncature explicitement
   signalée et **jamais présentée comme consensus complet**. La clarification
   parcourt une entrée à la fois et borne son estimation de mémoire de
   partition à1Mio ; coût d'I/O/calcul supplémentaire assumé.

**La garantie est conditionnelle.** Si la vraie transformation appartient à
la grammaire/profondeur énumérée, si les exemples sont exacts et si la recherche
est complète, elle reste parmi les hypothèses : un consensus lui correspond.
Sans ces hypothèses, aucun théorème de compréhension ni de sûreté générale.
Le système n'infère pas le français, ne découvre pas ses primitives et ne
connaît pas automatiquement les exceptions cachées.

## Expériences, dans leur ordre réel

### 1. Premier protocole fixé avant exécution — `results.json`

16 tâches internes : entiers, texte et listes ; une démonstration initiale,
7 entrées réservées par tâche, au plus3 clarifications, profondeur3. Une tâche
requiert4 opérations dans le langage fourni. La référence « programme le plus
court » choisit sans gérer l'ambiguïté. Les enseignants sont **simulés et exacts**.

- Plus-court programme initial :49/112 sorties correctes.
- Consensus initial :6/112 acceptées, toutes correctes ; couverture minuscule.
- Clarifications actives :103/112 correctes acceptées,0fausse,25questions.
- Questions aléatoires, seeds7/17/27 :252/336 correctes acceptées,0fausse,
  144questions. C'est la même suite répétée, pas336 tâches indépendantes.

Le pool fixe ne suffit pas à lever certaines ambiguïtés. `extended.py`,
**exploratoire après ce résultat**, génère64 candidats sans leurs labels et
exclut les collisions avec les entrées réservées. À budget maximal identique :

| Politique | Correctes acceptées /336 | Fausses acceptées | Questions |
|---|---:|---:|---:|
| Actif généré |314|0|56|
| Aléatoire généré |266|0|144|

L'actif s'arrête s'il ne trouve plus de désaccord ; l'aléatoire peut dépenser
les3 labels. Ce sont des budgets **maximaux** égaux, pas des nombres réalisés
égaux. Les exclusions de collisions n'utilisent aucune sortie de test.

### 2. Mauvaise classe de fonctions — résultats négatifs conservés

Dans `extended-results.json`, la vraie fonction `min(x,10)` est absente du
DSL. Après clarification,7 hypothèses sont d'accord et commettent3erreurs/7.
Une question d'audit `20→10`, qui ne vient pas du désaccord, les élimine toutes.

`open_world.py` étend ce contre-test à5 fonctions hors DSL ×3 seeds,13 tests
chacune. Expérience exploratoire, pas un benchmark externe :

| Politique (≤3 labels supplémentaires) | Réponses | Correctes | Fausses | Questions |
|---|---:|---:|---:|---:|
| Désaccord seul |117/195|84|33|15|
| 1 désaccord +2 audits de couverture |39/195|36|3|27|
| Aléatoire |39/195|36|3|31|

L'audit choisit une entrée numérique éloignée des observations, sans voir la
fonction réelle. **Il réduit les erreurs en détectant une inadéquation et en
refusant, pas en résolvant ces fonctions.** L'exception ponctuelle à137 reste
invisible aux questions posées : les3 erreurs persistent. Pas de sûreté ouverte.

### 3. Réouverture profondeur3→4 — `revision-results.json`

La tâche `strip → lower → split_comma → sort` avait zéro hypothèse à3 étapes.
L'élargissement automatique trouve20 hypothèses en11146 nœuds ; une seule
clarification/seed suffit ensuite :21/21 sorties correctes sur3 seeds,
2 syntaxes restantes, DB20480octets. Aucune nouvelle primitive ni phrase
spécifique ajoutée. Cela ne répare pas la classe incorrecte lorsque ses
hypothèses sont encore compatibles avec les observations disponibles.

### 4. Banc élargi à256 procédures générées — `generated-results.json`

Protocole dans `generated_bench.py` fixé avant son premier lancement :
seed20261019,256 syntaxes cibles distinctes avec domaine, longueurs1..4,
une démonstration initiale,≤3 labels supplémentaires,8 tests réservés par tâche.
Le générateur rejette les cibles invalides sur les entrées du banc : **sélection
favorable connue**, pas un test aveugle. Le solveur ne voit jamais les cibles.

| Cibles | Politique | Correctes acceptées | Fausses acceptées | Questions |
|---|---|---:|---:|---:|
|234 de longueur≤3,1872 tests|Actif|1872|0|238|
|idem|Aléatoire|1686|0|702|
|22 de longueur4,176 tests|Actif|128|8|19|
|idem|Aléatoire|110|0|57|

Total actif :2000 correctes,8fausses,40abstentions sur2048tests,257questions.
Aléatoire :1796 correctes,0fausse,252abstentions,759questions.
**L'actif améliore la couverture mais peut rendre les erreurs plus affirmées
quand sa grammaire est insuffisante.** Cas fautifs conservés :
`sort → take2 → sum → absolute` et `mul3 → sub1 → mul2 → absolute`.
Une cible écrite en4 étapes peut parfois être représentée plus court ; cette
catégorie n'affirme pas que toutes ses fonctions sont hors classe.

### 5. Coût de recherche — `scaling-results.json`

Processus frais par profondeur, un exemple `1→1`, mêmes plafonds :

| Profondeur | Nœuds | Hypothèses | Recherche | Clarification64 candidats | DB |
|---|---:|---:|---:|---:|---:|
|3|585|65|~0,005s|~0,024s|20480o|
|4|4681|384|~0,028s|~0,142s|57344o|
|5|37449|2279|~0,211s|~0,914s|278528o|
|6|100000, tronqué|3757|~0,555s|~1,567s|487424o|

À6 étapes : réponse `incomplete_search`, pas accord inventé. Cette croissance
exponentielle est un blocage réel ; un disque compact ne rend pas le calcul gratuit.
Un seul passage/configuration, pas d'intervalles de confiance ni d'énergie mesurée.

## Mémoire réellement mesurée

Linux/Python3.11.2, stdlib seule, CPU Xeon x86_64,2026-10-19. Aucun GPU,
aucun NumPy requis. `RLIMIT_AS=100000000` impose une limite **réelle d'espace
virtuel du processus** pour les benchmarks et le serveur. Ce n'est pas une
limite cgroup ni une mesure du cache noyau/OS, et pas une preuve de portabilité
à tous les appareils. Le module `resource` de ces outils cible Unix.

- Banc256 :14,50s, picRSS23834624octets (recherche et évaluation réunies).
- Échelle, processus frais :picsRSS~13,5–14,9Mo.
- Serveur :mesure `/proc` du **processus Python**, pas de son shell parent,
  enregistrée dans `server-results.json`. Environ21Mo RSS, limite100Mo vérifiée.
- SQLite :cache256Kio, mmap désactivé, temporaires sur fichier. Bases
  d'expériences supprimées après chaque run ; les résultats JSON sont conservés.

## Reproduction / tests

```sh
python -m unittest discover -s experiments/silex/induction -p 'test_*.py' -v
python experiments/silex/induction/bench.py > /tmp/s7.json
python experiments/silex/induction/extended.py > /tmp/s7-extended.json
python experiments/silex/induction/revision.py > /tmp/s7-revision.json
python experiments/silex/induction/open_world.py > /tmp/s7-open-world.json
python experiments/silex/induction/scaling.py > /tmp/s7-scaling.json
python experiments/silex/induction/generated_bench.py --output /tmp/s7-generated.json
# Après démarrage du serveur :
python experiments/silex/induction/smoke_server.py --pid <PID_PYTHON>
```

9 tests noyau passent : interpréteur fermé/borné, ambiguïté non effacée,
filtrage exact/persistance, contradiction et révision, troncature honnête,
budget de question, conservation de la vraie cible représentable, contre-exemple
au consensus hors classe, copies indépendantes/probes déterministes.
10 vérifications HTTP passent ; syntaxe JS vérifiée par `node --check`.
Les6 tests S6 et7 tests Paged passent toujours. Pas de test visuel navigateur
automatisé revendiqué. Le serveur est un labo local mono-requête, pas un produit
sécurisé multi-utilisateurs.8sessions maximum,16observations/session ; exemples
et hypothèses temporaires, suppression explicite disponible, pas de logs HTTP.

## Ce qui préexiste, ce que nous avons effectivement fait

La synthèse depuis exemples, les ensembles d'hypothèses et l'apprentissage actif
ne sont **pas des inventions de cette session**. Deux sources primaires lues :

- [DreamCoder, README épinglé](https://github.com/ellisk42/ec/blob/cb0e63f5c33cd2de360b791038b0f5272750270e/README.md) :
  algorithme wake-sleep découvrant des programmes, domaines texte/listes et
  outils d'apprentissage. Nous n'avons pas exécuté ses expériences ni reproduit
  son apprentissage de bibliothèque ou son réseau de reconnaissance.
- [Microsoft PROSE, README épinglé](https://github.com/microsoft/prose/blob/e04b29deac8ebda1be7edf9a7162d22bc29fbe62/README.md) :
  génération de programmes depuis exemples entrée/sortie et DSL écrits par
  les auteurs. Aucun code de ce SDK n'est intégré ici.

Notre apport à ce projet est un composant exécuté, borné en ressources,
interrogeable, avec résultats positifs ET contre-tests de confiance publiés.
Il peut apprendre des procédures nouvelles dans son langage, pas devenir seul
un interlocuteur généraliste. Prochaine question scientifique : comment élargir
ou apprendre la représentation sans explosion de recherche et sans confondre
accord interne et conformité à l'intention ? Pas d'autorisation de dépense requise
pour reproduire ces expériences.
