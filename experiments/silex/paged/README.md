# S5 — représentation sélective et bibliothèque sur disque

2026-10-19 · Silex · REVUE. **Banc synthétique de représentation/runtime, pas LLM.**
Cadrage utilisateur confirmé :100Mo RAM visés,200Mo acceptables ; disque permis.
Ni nombre de paramètres maximal ni taille de fichier artificiellement imposés.

## Hypothèse testée

S4 montrait qu'imposer une base partagée pouvait être moins bon que des facteurs
indépendants. Ici on apprend chaque opérateur sur des exemples, puis on choisit,
sur validation séparée, la plus petite représentation suffisamment fidèle :

- matrice indépendante (exception dense) ;
- identité + résidu de faible rang indépendant ;
- identité + combinaison d'une base commune, si le coût GLOBAL de cette base
  est amorti par les économies et si le seuil de qualité est respecté.

Les exceptions restent sur disque ; seules celles demandées entrent dans un
cache LRU borné. **L'adresse de l'opérateur est fournie par le banc. Aucun
routeur linguistique n'est appris.** Pas de réponse textuelle générée ; pas de
récupération de faits, ni preuve de qualité conversationnelle.

## Protocole

Dimension32,192 puis1536 opérateurs ; 1/3 construits avec quatre facteurs communs,
1/3 échanges de positions,1/3 permutations arbitraires. Cette structure est
**plantée** dans les données synthétiques, pas découverte dans de vrais poids.
Le constructeur connaît la vérité pour créer les exemples et évaluer, mais le
sélecteur de format n'accède ni aux labels de familles ni aux vecteurs de test.

Chaque opérateur est appris par moindres carrés sur64 vecteurs gaussiens. Choix du
format sur96 AUTRES vecteurs, erreur relative maximale acceptée1e-4. Test sur7
nouveaux vecteurs par opérateur. Les identités des opérateurs testés sont déjà
apprises : généralisation à de nouvelles entrées, pas à de nouvelles règles.

Base commune globale : SVD des résidus appris aplatis. Variante groupée : comparer
les projecteurs des espaces lignes/colonnes estimés sur l'entraînement ; apprendre
une base par groupe candidat et retenir le groupe donnant la meilleure économie
nette validée. Un seul dictionnaire résident dans cette version ; autres opérateurs
indépendants. Mécanismes connus, **pas de revendication d'invention**.

SQLite stocke des BLOB binaires float64 (en-tête inclus dans le décompte). Cache
SQLite256Kio, mmap désactivé. Budget du cache applicatif64/256Kio : base partagée
résidente DÉDUITE du budget, pas ajoutée gratuitement. Ce budget ne comprend PAS
objets Python, pages SQLite, buffers temporaires ni page cache OS ; RSS mesuré à part.

## Résultats en quatre boucles

### 1. Partage global refusé automatiquement

192 opérateurs : le dictionnaire global ne tient pas le seuil/coût ; désactivé.
Les résidus indépendants donnent689088octets de payload contre1573824 pour dense.
Accès Zipf simulés/cache64Kio :14,88Mo de BLOB relus pour dense contre4,49Mo pour
adaptatif (3000 requêtes). Cache OS chaud, lectures physiques nulles : ne pas
interpréter les temps chauds comme un benchmark SSD.

### 2. Partage par groupes utile, mais pas toujours pour le cache

192 opérateurs :64codes partagés,64résidus indépendants,64exceptions denses.
Base commune32768octets, payloads560064octets (hors base) vs689088 indépendants.
Mais au budget64Kio, la base occupe la moitié du cache : accès Zipf,4,64Mo de
BLOB relus vs4,49Mo pour indépendant. **Une meilleure compression de stockage
peut dégrader le working set.** Au budget256Kio, hybride1,29Mo vs1,79Mo.

### 3. Bruit d'apprentissage : fragilité et contrôle

Bruit gaussien sigma1e-5 ajouté aux sorties d'entraînement, pas aux vérités de
validation/test. Groupement strict1e-6 : le dictionnaire est désactivé. Tolérance
sur projecteurs1e-3 : regroupement retrouvé ; qualité tenue à l'écart reste sous
1,21e-5 d'erreur relative. Ce réglage est exploratoire, pas une garantie sur d'autres
niveaux de bruit/distributions. Aucun «seuil universel» n'est déduit de ce cas.

### 4. Bibliothèque1536, lectures réellement comptées

| Stockage | Dense | Adaptatif indépendant | Hybride groupé |
|---|---:|---:|---:|
| Payloads BLOB (octets) | 12590592 | 5512704 | 4480512 |
| Fichier SQLite (octets) | 13377536 | 6299648 | 4804608 |
| Base commune, fichier (octets) | 0 | 0 | 32896 |

Ajouter aussi le petit metadata.json commun si l'on compte la totalité des fichiers.
La bibliothèque sur disque est reconstruite, pas versionnée.

DONTNEED sur le fichier DB de CETTE expérience uniquement, avant ouverture, sans
purge globale. C'est une recommandation au système, pas un état froid contrôlé du
disque physique. 3000 accès Zipf(alpha1,2), cache applicatif64Kio, un passage :

| Mesure | Dense | Indépendant | Hybride |
|---|---:|---:|---:|
| BLOB lus au total | 17057957o | 5757153o | 5329507o |
| read_bytes processus, delta | 8589312o | 4730880o | 3538944o |
| p95 par application d'opérateur | 0,344ms | 0,140ms | 0,136ms |
| RSS pic du banc | 37,14Mo | 37,10Mo | 37,88Mo |

`read_bytes` inclut les effets de lecture anticipée et granularité du système ;
ce n'est pas la somme des seuls poids utiles. Le chargement initial/ouverture
SQLite n'est pas dans l'intervalle chronométré. Une exécution par configuration,
pas d'intervalle de confiance ni de conclusion robuste sur les petits écarts
entre méthodes compactes. Les tests numériques passent sur ce banc.

**Avertissement mémoire :** cgroup `/user` partagé~326–339Mo dans ces runs,
au-dessus200Mo, alors que RSS processus~37–38Mo. Il inclut d'autres processus,
cache fichiers et environnement. On ne peut ni attribuer ce total à ce modèle,
ni prétendre avoir prouvé l'objectif200Mo tout compris. Le cgroup est en lecture
seule pour cet utilisateur ; aucune limite200Mo n'a été appliquée. Les premiers
JSON portent `null` pour cgroup (ancien lecteur ne résolvant pas `/user`) ; les
runs `large-cold-file-results.json` utilisent le chemin corrigé et indiquent la
mesure après chronométrage ET après la passe qualité séparément.

## Reproduction

Depuis la racine, NumPy2.2.6/Python3.11.2, CPU Xeon2,60GHz x86_64/2vCPU KVM,
Linux, OPENBLAS_NUM_THREADS=1. Date2026-10-19. Ni GPU ni capteur énergétique.
Les octets de bibliothèque/cache ne sont pas des paramètres d'un généraliste.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/paged/test_library.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/paged/bench.py build
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/paged/run_suite.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/paged/bench.py build --group-dictionary --root .cache/silex-s5-group
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/paged/run_suite.py --root .cache/silex-s5-group --output experiments/silex/paged/group-runtime-results.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/paged/bench.py build --group-dictionary --noise 0.00001 --group-tolerance 0.001 --root .cache/silex-s5-noise-relaxed
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/paged/run_suite.py --root .cache/silex-s5-noise-relaxed --output experiments/silex/paged/noise-runtime-results.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/paged/bench.py build --group-dictionary --count 1536 --root .cache/silex-s5-large
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/paged/run_suite.py --root .cache/silex-s5-large --output experiments/silex/paged/large-runtime-results.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/silex/paged/run_suite.py --root .cache/silex-s5-large --cold-file --output experiments/silex/paged/large-cold-file-results.json
```

Pour bruit/groupement strict : omettre `--group-tolerance 0.001` et utiliser un
root distinct ; ne pas écraser un dossier qu'un autre run lit. `build` remplace
les trois DB, basis.npy et metadata.json du root choisi. Les fichiers .cache
sont ignorés et régénérables. Résultats JSON légers versionnés.

Sept tests : codecs, inférence sur entrées nouvelles, budget/hit/éviction LRU,
BLOB trop gros non mis en cache, base résidente comptée, DB lecture seule/ID
inconnu, déterminisme/comptabilité disque. Les noms internes `shared` ne signifient
pas qu'une base est active : lire `shared_enabled` dans le résultat de build.

## Ce qui manque avant une architecture de langage

1. Remplacer les IDs fournis par un contrôleur appris et mesurer ses erreurs,
   ses lectures supplémentaires et ses coûts. Ne pas lui attribuer l'oracle.
2. Répéter sur matrices réellement apprises dans un modèle de langage ; les
   facteurs communs synthétiques ne prouvent pas leur existence dans ces poids.
3. Choisir le format sur coût TOTAL observé (latence, RAM, disque), pas seulement
   taille de fichier : la boucle2 a déjà réfuté ce raccourci.
4. Mesure isolée sous plafond mémoire imposé, plusieurs répétitions et matériel
   cible ; formation des groupes scalable, actuellement coûteuse à l'entraînement.
5. Qualité de langage et énergie par réponse utile encore NON mesurées.
