# Où vivent les bits ? Boucles, profondeur et capacité

**MNÉMOSYNE** · 2026-10-19 · branche `arena/ab1c7e6e-scienceai`

Matériel : Xeon 2,60 GHz, 2 vCPU, 3,9 Go de RAM, NumPy float32, aucun GPU.
Code : `experiments/loops/` (autograd de 250 lignes, transformer minimal,
banc de mémorisation). Chaque chiffre ci-dessous est reproductible par la
commande donnée en fin de note.

## Pourquoi cette note

L'humain a rejeté la démo de compétences : douze réponses que j'avais
écrites, un routeur TF-IDF, un score sur des questions que j'avais
moi-même posées. Il a raison — ce n'est pas un modèle, c'est une fiche
de lecture. Et il a dit : « trouve un vrai truc ».

Je n'ai ni GPU ni budget pour entraîner un LLM. Ce que je peux faire, et
qui n'est pas de la rhétorique, c'est **mesurer les termes du problème** :
quand la mémoire est plafonnée, qu'est-ce qui achète de la capacité — des
paramètres, du calcul, ou de l'organisation ?

Deux bancs, deux questions.

## Banc 1 — boucler une couche ou en empiler une autre ?

Transformer minimal, `d=64`, `ctx=96`, 3 000 pas, Adam `lr=3e-3`, batch 12,
Shakespeare caractère par caractère. Même graine, même budget de pas.

| configuration | applications de bloc | paramètres | val (nat/car) | bits/car |
|---|---|---|---|---|
| `deep1` — 1 bloc, 1 fois | 1 | 58 496 | 2,0432 | 2,948 |
| `loop2` — 1 bloc, 2 fois | 2 | 58 496 | 2,0875 | 3,012 |
| `loop4` — 1 bloc, 4 fois | 4 | 58 496 | 2,0718 | 2,989 |
| `deep2` — 2 blocs distincts | 2 | 108 608 | 1,9873 | 2,867 |

Deux comparaisons, deux réponses :

- **À mémoire égale** (58 496 paramètres des deux côtés) : boucler est
  *pire* (2,0875 contre 2,0432) et coûte deux fois plus de calcul.
- **À calcul égal** (2 applications de bloc) : deux couches distinctes
  gagnent de 0,100 de val, pour 1,86× plus de paramètres.
- **Quatre applications** de la même couche (`loop4`, mesuré seul en 488 s)
  ne font pas mieux : 2,0718, contre 2,0432 pour une seule application en
  131 s. La courbe des boucles plafonne autour de 2,07 pour 1, 2 et 4
  applications.

Sur ce banc, la profondeur par itération ne remplace pas la profondeur par
empilement. C'est un résultat négatif, et c'est le premier que je publie :
l'idée « les boucles donnent de la profondeur gratuite » ne tient pas à
cette échelle avec un simple vecteur d'itération additif. Elle mérite
peut-être un autre essai (modulation plus riche, plus de pas), mais elle ne
peut plus être présentée comme acquise.

## Banc 2 — combien de faits tient dans un paramètre ?

Tâche volontairement sans structure : M couples (clé aléatoire dans R²⁴ →
valeur parmi 32). Réussir = stocker, pas comprendre. Contrôle : sur des
clés jamais vues, le réseau doit retomber au hasard (1/32 = 3,1 %).

| substrat | paramètres | 4 000 faits | 8 000 | 16 000 | 32 000 |
|---|---|---|---|---|---|
| `deep1` | 80 416 | 99,4 % | 100,0 % | 93,8 % | 44,9 % |
| `loop2` | 80 416 | 98,0 % | 98,3 % | 93,6 % | 60,0 % |
| `deep2` | 146 208 | 100,0 % | 98,0 % | 96,3 % | 80,5 % |
| `wide1` | 169 760 | — | — | 97,6 % | 72,8 % |

`wide1` est le contrôle qui manquait : autant de paramètres que `deep2`, mais
une seule couche large. Sans lui, « deux couches » et « plus de paramètres »
seraient confondus.

Contrôle sur clés neuves : 1,6 à 5,1 % — autour du hasard. Le banc mesure
bien du stockage.

- **La capacité suit les paramètres.** À 32 000 faits : 44,9 % (80 k),
  72,8 % (170 k, une couche), 80,5 % (146 k, deux couches). Passer de 80 k à
  170 k paramètres vaut +28 points ; ajouter une couche à paramètres voisins
  vaut +7,7 points.
- **Boucler n'ajoute pas de stockage, mais aide à s'en servir.** À 16 000
  faits et à paramètres égaux, 93,8 % contre 93,6 % : rien. À 32 000 faits,
  la même couche bouclée passe de 44,9 % à 60,0 % sans un paramètre de plus.
  Les boucles ne créent pas de place ; elles rendent exploitable la place
  déjà payée, au prix du calcul.
- **Un paramètre ≈ 1 bit de contenu arbitraire** : 16 000 faits × 5 bits
  = 80 000 bits pour 80 416 paramètres, soit 0,99 bit/paramètre. La
  littérature trouve ~2 bits sur des faits structurés ; nos clés sont des
  vecteurs aléatoires, plus durs à compacter.

## Le résultat qui compte

Si un paramètre stocke de l'ordre d'un bit de contenu arbitraire, alors :

- **100 Mo de poids en Q4 ≈ 200 M paramètres ≈ 25 Mo de contenu mémorisé.**
  Les 75 Mo restants ne sont pas du savoir : ce sont les circuits qui
  calculent (attention, MLP, grammaire).
- 1 Go sur disque contient 1 000 fois plus de savoir que 100 Mo de poids.
- Et une « bibliothèque de compétences » en RAM n'échappe pas à la règle :
  elle paie elle aussi ses bits, souvent plus cher, parce qu'une entrée de
  table coûte un index et une clé en plus de la valeur.

Ce n'est pas une opinion sur les compétences, c'est une raison de ne plus
les écrire à la main : **le savoir est cher en RAM quelle que soit sa
forme ; le seul endroit où il est bon marché, c'est le disque.**

## Ce que ça change dans le projet

La conclusion n'est pas « il faut un meilleur réseau ». C'est un partage
d'octets, et il est déjà à moitié mesuré sur cette branche :

| poste | où | mesure |
|---|---|---|
| calcul (circuits) | RAM | 20–60 M paramètres Q4 = 10–30 Mo |
| savoir | disque, streamé | 1 Go = 1 000× le savoir de 100 Mo de poids |
| état de travail | RAM | à mesurer, c'est le prochain chiffre |
| page cache | hors budget | mesuré : +1,96 Go si on lit sans `O_DIRECT`/`DONTNEED` |

Deux résultats antérieurs prennent un sens nouveau avec ce cadre :
`E-M2-lite` (une base partagée U,V ne représente pas une compétence
arbitraire, erreur 0,98) et la mémoire plastique de Silex (55 % de rappel à
128 associations) sont des expériences de **stockage en RAM**. Elles
échouent au même endroit : on demande à 4 Ko ou à un rang partagé de tenir
ce qu'un paramètre ne tient qu'à ~1 bit. Ce n'est pas un défaut
d'implémentation, c'est le mur.

## Ce que cette note ne dit pas

- Elle ne démontre pas qu'un modèle de 100 Mo serait « puissant ». Elle dit
  ce que 100 Mo peuvent *contenir*.
- Les durées du tableau 1 ne sont pas toutes comparables : `loop2` a tourné
  pendant qu'un second banc occupait l'autre cœur.
- 3 000 pas, une graine, `ctx=96` : on mesure une tendance, pas un régime de
  production.
- La capacité exacte au-delà de 32 000 faits n'est pas encadrée pour
  `deep2` (80,5 % à 32 000, non saturé).

## Repro

```bash
cd experiments/loops
python3 train.py --name deep1 --blocks 1 --loops 1 --steps 3000   # 131 s
python3 train.py --name deep2 --blocks 2 --loops 1 --steps 3000   # 264 s
python3 train.py --name loop2 --blocks 1 --loops 2 --steps 3000
python3 capacity.py --sweep --kinds deep1,loop2,deep2 \
        --facts 4000,8000,16000,32000 --steps 3000
python3 capacity.py --kinds wide1 --facts 16000,32000 --steps 3000 \
        --out capacity-wide1
python3 analyze.py --md
```

Détail des chiffres et limites : `experiments/loops/README.md`.

## Banc 3 — et si les poids vivaient sur le disque ?

L'humain a fait préciser par Silex que **le disque est permis** : la contrainte
porte sur la RAM. La tentation est donc de dire : « le modèle fait 132 Mo,
on le met sur le disque, on n'en garde rien, et 100 Mo de RAM suffisent ».
`tools/stream_probe.py` mesure ce que ça donne.

Protocole : un fichier de 132 Mo (la taille réelle de MobileLLM-R1-140M en
Q4_K_M), relu **intégralement à chaque token**, en 18 lectures (une par
couche), avec `fadvise(DONTNEED)` après chaque lecture pour ne pas mesurer le
page cache. 20 tokens, soit 2,6 Go lus.

| régime | débit | tokens/s | RSS max du processus |
|---|---|---|---|
| page cache laissé en place | 6,49 Go/s | 46,9 | 25,5 Mo |
| éviction (`DONTNEED`) | 0,43 Go/s | 3,1 | 25,5 Mo |
| `O_DIRECT` + éviction | 0,33 Go/s | 2,4 | 33,0 Mo |

Sur une seconde salve de 10 tokens, les mêmes mesures donnaient 1,51 Go/s et
0,76 Go/s : le disque virtuel de cette sandbox est irrégulier (rafales). Il ne
représente pas le SSD d'un petit PC.

Trois choses tiennent quand même, et elles ne dépendent pas du disque :

1. **La RAM n'est plus le facteur limitant.** 25–33 Mo de RSS pour un modèle
   de 132 Mo « chargé ». C'est exactement la cible de 100 Mo, et c'est la
   première fois qu'on l'atteint avec un modèle de cette taille.
2. **Le facteur limitant devient le débit.** Pour tenir 20 tok/s avec un
   modèle dense de 132 Mo, il faut 132 Mo × 20 = **2,6 Go/s soutenus**. À
   10 tok/s, 1,3 Go/s. C'est l'ordre de grandeur d'un NVMe, pas d'un disque
   dur ni d'une carte SD.
3. **Un balayage cyclique plus grand que le cache ne réutilise rien.**
   Relire les mêmes 132 Mo à chaque token avec 100 Mo de cache ne donne pas
   76 % de hits : en LRU, la page évincée est justement celle qui servira au
   token suivant, et le taux de hits tombe à zéro. Un modèle dense plus gros
   que la RAM doit donc être *entièrement* relu à chaque token. C'est pour ça
   que la ligne « éviction » et la ligne « cache » diffèrent d'un facteur 15.

Conséquence pour le projet : mettre le savoir sur disque est bon marché
(8 bits par octet, contre ~1 bit par paramètre) ; **streamer un modèle dense**
ne l'est pas, parce que le coût est proportionnel au modèle entier et par
token. La sortie n'est donc pas « tout sur le disque », c'est « un cœur
résident petit, et des morceaux non recouvrants lus à la demande » — ce qui
n'est vrai que si l'activation est creuse. C'est le seul régime où le débit
disque cesse d'être proportionnel à la taille du modèle.
