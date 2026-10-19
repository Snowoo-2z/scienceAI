# Boucles vs couches — un banc pour mesurer ce qui achète de la capacité

Tout est en NumPy, sur ce CPU, sans GPU. Le but n'est pas de produire un
modèle utilisable (58 k paramètres ne parlent pas) mais de **mesurer** ce que
coûtent la profondeur et le stockage quand la mémoire est la contrainte.

Deux bancs indépendants :

| fichier | question | sortie |
|---|---|---|
| `train.py` | à mémoire égale / à calcul égal, boucler une couche vaut-il mieux que l'empiler ? | `results/loops/*.json` |
| `capacity.py` | combien de faits arbitraires tient dans un paramètre ? | `results/loops/capacity.json` |
| `analyze.py` | met les résultats en tableau | texte ou markdown |

## Reproduire

```bash
cd experiments/loops
python3 train.py --name deep1 --blocks 1 --loops 1 --steps 3000
python3 train.py --name deep2 --blocks 2 --loops 1 --steps 3000
python3 train.py --name loop2 --blocks 1 --loops 2 --steps 3000
python3 train.py --sweep --steps 3000          # toutes les configurations
python3 capacity.py --sweep --kinds deep1,loop2,deep2 \
        --facts 4000,8000,16000,32000 --steps 3000
python3 capacity.py --kinds wide1 --facts 16000,32000 --steps 3000 --out capacity-wide1
python3 analyze.py --md
```

Corpus : `data/input.txt`, Shakespeare caractère par caractère
(1 115 394 octets, 65 symboles), copié depuis
`https://github.com/karpathy/char-rnn` (`data/tinyshakespeare/input.txt`,
domaine public). Un caractère = un token ; la perte en bits/caractère se
compare directement à 4,7 bits pour un tirage uniforme.

## Résultats — banc texte

<!-- analyse: début -->
| configuration | blocs | boucles | applications | paramètres | val | bits/car | secondes |
|---|---|---|---|---|---|---|---|
| deep1 | 1 | 1 | 1 | 58 496 | 2.0432 | 2.948 | 131 |
| deep2 | 2 | 1 | 2 | 108 608 | 1.9873 | 2.867 | 264 |
| loop2 | 1 | 2 | 2 | 58 496 | 2.0875 | 3.012 | 593* |
| loop4 | 1 | 4 | 4 | 58 496 | 2.0718 | 2.989 | 488 |
<!-- analyse: fin -->

`*` 593 s : `loop2` a tourné pendant qu'un autre banc occupait le second
cœur ; à comparer avec `deep2` (264 s) seulement après une mesure à vide.
`loop4` : même couche appliquée 4 fois, mesuré seul (488 s).

Lecture, à ce stade :

- **à calcul égal** (2 applications de bloc), `deep2` bat `loop2` de 0,100 de
  val avec 1,86× plus de paramètres ;
- **à mémoire égale** (58 496 paramètres), `loop2` est *moins bon* que
  `deep1` de 0,044 tout en coûtant deux fois plus de calcul ;
- **quatre applications** de la même couche (`loop4`, 488 s seul) donnent
  2,0718 : toujours au-dessus de `deep1` (2,0432, 131 s). Boucler ne
  rattrape pas ce que les paramètres apportent ; la courbe plafonne autour
  de 2,07 pour 1, 2 et 4 applications.

Autrement dit, sur ce banc et à cette échelle : **le calcul partagé ne
remplace pas les paramètres**. C'est un résultat négatif pour l'hypothèse
« profondeur gratuite par itération », et il est reproductible en 131 s.

## Résultats — banc capacité

M = nombre de couples (clé aléatoire dans R²⁴ → valeur parmi 32). Les clés de
test sont neuves : la colonne `test` mesure la généralisation, qui doit rester
au hasard (1/32 = 3,1 %) — sinon le banc mesure autre chose que du stockage.

| substrat | paramètres | couches | 4 000 | 8 000 | 16 000 | 32 000 |
|---|---|---|---|---|---|---|
| `deep1` | 80 416 | 1 | 99,4 % | 100,0 % | 93,8 % | 44,9 % |
| `loop2` | 80 416 | 1 bouclée 2× | 98,0 % | 98,3 % | 93,6 % | **60,0 %** |
| `deep2` | 146 208 | 2 | 100,0 % | 98,0 % | 96,3 % | **80,5 %** |
| `wide1` | 169 760 | 1, large | — | — | 97,6 % | 72,8 % |

`wide1` est le contrôle qui manquait : mêmes paramètres que `deep2` (146 k
contre 170 k) mais une seule couche large. Il sépare « plus de paramètres »
de « plus de profondeur ».

`test` sur clés neuves : 1,6–5,1 % (hasard = 3,1 %) → le banc mesure bien du
stockage, pas de la généralisation.

Quatre lectures :

1. **La capacité suit les paramètres.** À 32 000 faits : 44,9 % (80 k), 60,0 %
   (80 k bouclé), 72,8 % (170 k, une couche), 80,5 % (146 k, deux couches).
2. **Boucler n'ajoute pas de stockage, mais aide à exploiter celui qu'on a.**
   À 16 000 faits, boucler ne change rien (93,8 % → 93,6 %) : ce qui est
   stockable l'est déjà. À 32 000 faits, la même couche bouclée passe de
   44,9 % à 60,0 % sans un paramètre de plus — +15 points gratuits en RAM,
   payés en calcul.
3. **La profondeur non plus n'est pas le facteur principal.** À paramètres
   voisins, `deep2` (146 k, deux couches) fait 80,5 % et `wide1` (170 k, une
   couche) 72,8 % : +7,7 points pour la profondeur, contre +28 points pour
   les paramètres quand on passe de 80 k à 170 k.
4. **Un paramètre ≈ 1 bit de contenu arbitraire.** 16 000 faits × 5 bits
   = 80 000 bits pour 80 416 paramètres. La littérature mesure ~2 bits/param
   sur des faits structurés ; ici les clés sont des vecteurs aléatoires dans
   R²⁴, donc plus durs à compacter, et on obtient 0,99.

Conséquence arithmétique, à prendre comme telle : **100 Mo de poids en Q4
= ~200 M paramètres = ~25 Mo de contenu mémorisé**, le reste des octets sert
au calcul (attention, MLP, grammaire), pas au savoir. Un gigaoctet de disque
en contient 1 000 fois plus.

## Limites

- 3 000 pas, un seul jeu, une seule graine ; les écarts < 0,02 de val ne
  veulent rien dire.
- `loop_id` (le signal d'itération) est un simple vecteur additif ; une
  modulation plus riche changerait peut-être la conclusion.
- Les durées ne sont comparables qu'entre mesures faites seules.
- `ctx = 96`, `d = 64` : on mesure une *tendance*, pas un régime de production.

## Notes d'implémentation

- `autograd.py` : autograd de ~250 lignes, 16 opérations, gradient checké
  (erreur relative 3,8e-05 à 2,1e-04).
- Le graphe contient un cycle (parent → enfant par `kids`, enfant → parent
  par la fermeture `back`). Sans nettoyage explicite en fin de `backward`,
  seuls les passages du ramasse-miettes libèrent la mémoire : constaté
  2,24 Go de RSS pour un modèle de 58 k paramètres au pas 2 100. `backward`
  coupe maintenant les arêtes ; le RSS reste plat.
