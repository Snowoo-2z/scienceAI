# 08 — Mesurer la demande avant de construire quoi que ce soit

**2026-10-19 · KAIROS · branche `arena/c9070fe4-scienceai`**
Outil : `tools/zipf_check.py` · sorties : `results/zipf-demo.txt`

---

## 1. Pourquoi ce document existe

`docs/06-PROGRAMME-II.md` §10 montre que le rendement du tri-système dépend
de deux nombres, et de deux seulement :

1. **α**, l'exposant de Zipf de la demande réelle de compétences ;
2. **k**, le nombre moyen de décompositions alternatives par requête.

Avec α = 1 et une décomposition, 10 M de compétences couvrent 78 % des
requêtes. Avec α = 0,7, la même bibliothèque n'en couvre que 25 % et
l'avantage tombe de 23× à 1,5×. Avec trois décompositions, la couverture
effective monte à 99 %.

Ces deux nombres **ne se devinent pas**. Ils se mesurent sur des logs
d'usage. C'est la première expérience du programme, elle coûte zéro euro,
et elle peut l'arrêter avant qu'on dépense les 5 000 € de E-Ω5.

## 2. Ce qu'il faut journaliser

Une ligne JSONL par requête, avec quatre champs utiles :

```json
{"ts": "2026-10-19T14:03:11Z", "label": "resumer_article",
 "alternatives": ["resumer_article", "extraire_points"], "ok": true}
```

- `label` : la compétence **réellement nécessaire**, décidée après coup
  (par l'utilisateur ou par un annotateur), pas celle que le routeur a
  choisie. C'est le seul champ qui compte pour α.
- `alternatives` : les autres compétences qui auraient aussi résolu la
  requête. Facultatif ; c'est ce qui donne k.
- `ok` : la réponse a-t-elle été acceptée. Sert à vérifier que la couverture
  mesurée n'est pas achetée par des réponses fausses.
- `ts` : pour suivre la dérive de la distribution dans le temps.

**Ne pas journaliser le texte des requêtes** si ce n'est pas nécessaire :
un label suffit, et il évite les problèmes de données personnelles. Si le
texte est nécessaire pour annoter, le hacher et ne garder que le label.

Taille minimale : **10 000 requêtes** sur au moins deux semaines. En dessous,
l'estimation de α a un intervalle de confiance trop large pour trancher.
Pour k, 200 requêtes annotées en double suffisent (accord inter-annotateurs
à publier).

## 3. Comment lancer la mesure

```bash
python3 tools/zipf_check.py logs.jsonl
python3 tools/zipf_check.py logs.jsonl --json > results/zipf-reel.json
python3 tools/zipf_check.py --self-test
```

L'outil estime α par maximum de vraisemblance d'une loi de puissance
tronquée, puis calcule :

- la couverture empirique du top-n (mesurée, sans modèle) ;
- le nombre de compétences nécessaires pour 90 / 95 / 99 % (modèle ajusté) ;
- la moyenne de k et la couverture effective `1 − (1 − c)^k`.

L'auto-test vérifie que l'estimateur retrouve α à 0,02 près sur des
échantillons synthétiques (α ∈ {0,7 ; 1,0 ; 1,3}).

## 4. Règle de décision, écrite à l'avance

| α mesuré | Lecture | Action |
|---|---|---|
| **< 0,8** | longue traîne trop lourde ; la bibliothèque ne couvre presque rien | **pivoter** : chercher une autre invention, ne pas construire la bibliothèque |
| **0,8 – 1,2** | zone de travail ; l'avantage dépend de k | mesurer k sur 200 requêtes annotées, puis E-Ω1 |
| **> 1,2** | tête très concentrée ; la bibliothèque est très rentable | aller à E-Ω1 directement |

Seuils fixés avant la mesure pour éviter de les ajuster après coup.

## 5. Le piège de l'estimation

Les labels rares non observés sont absents des logs : α est donc **sous-estimé**
si l'on prend M = nombre de labels observés. Deux parades :

1. publier aussi la couverture **empirique** du top-n, qui ne dépend
   d'aucun modèle ;
2. si l'on veut le modèle, fixer M explicitement (par exemple M = 10⁹
   procédures possibles dans `tools/skill_econ.py`) et le justifier.

C'est la raison pour laquelle `tools/zipf_check.py` affiche les deux :
la couverture mesurée et la prédiction ajustée. Si les deux divergent de
plus de 5 points, le modèle de Zipf n'est pas le bon et il faut le dire.

## 6. Démonstration sur données synthétiques

`results/zipf-demo.txt` contient trois jeux de 200 000 requêtes, 10 000 labels,
générés avec α = 0,7 / 1,0 / 1,3. L'estimateur retrouve 0,705 / 1,000 / 1,280.

| α réel | couverture top-1 000 | skills pour 90 % |
|---:|---:|---:|
| 0,7 | 47,7 % | 7 141 |
| 1,0 | 76,6 % | 3 646 |
| 1,3 | 94,5 % | 314 |

Lecture : à α = 0,7, il faut 7 141 compétences pour couvrir 90 % de la
demande ; à α = 1,3, 314 suffisent — un facteur 23 sur la taille de la
bibliothèque, donc sur son coût de construction. C'est le paramètre le plus
lourd du programme après la qualité du vérificateur.

## 7. Ce que cette mesure ne dit pas

- Elle ne dit rien de la **qualité** des compétences : une couverture élevée
  avec 30 % de réponses fausses est pire qu'une couverture faible et honnête.
- Elle ne dit rien du **coût de construction** d'une compétence (extraction,
  test, déduplication). Si ce coût dépasse 1 € par compétence, 10 M de
  compétences coûtent 10 M€ et le programme est mort.
- Elle suppose que la demande est **stationnaire**. Un assistant de site web
  voit sa distribution changer avec les utilisateurs ; il faut la remesurer
  tous les mois.

Malgré ces limites, c'est la mesure la moins chère et la plus décisive du
programme : **deux nombres, zéro euro, et la réponse est écrite d'avance**.
