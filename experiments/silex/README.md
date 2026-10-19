# S2 — opérateurs partagés sans matérialisation

**Prototype mécanique, pas modèle de langage.** 2026-10-19, Silex/W3.
Le cadrage utilisateur confirme 100 Mo et une ambition généraliste ; aucun
plafond de paramètres n'est imposé. Une factorisation bas-rang n'est pas une
invention nouvelle ; cette expérience vérifie une brique, pas une percée.

## Reproduire (Linux/macOS, Python 3, aucune dépendance)

```sh
python experiments/silex/shared_operators.py --self-test
python experiments/silex/shared_operators.py --steps 8
python experiments/silex/shared_operators.py --steps 128
python experiments/silex/shared_operators.py --steps 2048
```

Calcul : `W_i = U diag(c(i)) V`, exécuté sans construire `W_i`.
État : `x <- tanh(x + W_i x)`. Le code c(i) est une sinusoïde déterministe,
ni apprise ni porteuse de connaissances. U et V sont initialisés aléatoirement.

## Mesures du sandbox

CPU exposé : Intel Xeon @ 2,60 GHz, 2 vCPU KVM, x86_64 ; Python 3.11.2,
Linux 6.1.158+. Float64 sans quantification ; un vecteur synthétique de dimension
64 ; rang 8 ; seed 42 ; processus indépendant pour chaque configuration.

| Étapes | Charge utile des facteurs | Pic RSS processus | Initialisation + exécution |
|---:|---:|---:|---:|
| 8 | 8 192 octets | 11 571 200 octets | 0,00245 s |
| 128 | 8 192 octets | 11 653 120 octets | 0,01535 s |
| 2048 | 8 192 octets | 11 653 120 octets | 0,23186 s |

Un passage par configuration, pas d'intervalle de confiance. RSS via getrusage ;
ni mémoire système/cgroup, ni énergie mesurées. Le nombre d'étapes n'est pas un
nombre de tokens et ne mesure pas le débit d'un LLM. Neuf vérifications contre
une matrice explicitement construite passent avec erreur absolue < 1e-12.
Les résultats bruts sont les trois fichiers result-*.json.

**Ce qui est vérifié :** cette implémentation compose des opérateurs sans stocker
une matrice indépendante par étape ; le temps augmente, les facteurs restent fixes.
La colonne JSON hypothetical_independent_dense_bytes est un calcul de stockage
hypothétique, PAS une comparaison mémoire mesurée ni une équivalence de capacité.

**Ce qui n'est pas vérifié :** apprentissage, qualité conversationnelle, mémoire
à long terme, généralisation, entraînement sous 1 Go, portabilité sur téléphone,
consommation, passage à l'échelle ou supériorité sur une autre architecture.

## Limite qui oriente la prochaine idée

Toutes les matrices W_i sont dans un espace engendré par au plus r matrices
`u_j v_j^T`. Multiplier les identifiants ne multiplie donc pas la capacité de
stockage indépendante. La non-linéarité et la composition ajoutent de la
profondeur de calcul, pas gratuitement de nouveaux faits appris.

**Piste expérimentale, non implémentée :** générer c à partir de l'état courant
via un contrôleur appris, composer des opérateurs avec permutations structurées,
et choisir le nombre de récurrences selon la difficulté. Cela associerait poids
partagés, état récurrent et calcul adaptatif plutôt qu'un empilement Transformer
classique. Aucun caractère inédit n'est revendiqué sans revue de littérature.

## S3 proposé : vérifier la capacité, pas seulement la mémoire

1. Même budget de paramètres stockés, données, optimisation et calcul mesuré :
   comparer récurrence partagée fixe, opérateurs modulés par l'état, et référence
   à couches indépendantes (ajuster largeur pour égaliser le stockage).
2. Apprendre rappel associatif, copie retardée et composition de règles ; tenir
   des longueurs et combinaisons à l'écart de l'entraînement. Plusieurs seeds.
3. Mesurer exactitude hors distribution, perte, temps, RAM, nombre d'étapes ;
   rechercher effondrement de l'état et saturation de capacité.
4. Puis seulement langage : corpus traçable avec séparation train/test, perte
   sur texte général, tests conversationnels variés, comparaison à budget égal.
   Une réussite synthétique ne démontre pas la conversation générale.
5. Si la modulation n'apporte pas de qualité à coût comparable, revoir ce
   mécanisme plutôt que grossir le compte de « paramètres virtuels ».
