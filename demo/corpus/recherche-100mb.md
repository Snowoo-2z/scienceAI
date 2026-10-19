# Project LLM — Concevoir un LLM utile sous 100 Mo de RAM

**Version** : 0.1 — 2026-10-19
**Auteur** : Agent de recherche (branche `arena/9c2ae177-scienceai`)
**Statut** : rapport de cadrage + état de l'art + architecture candidate + plan d'expériences
**Périmètre** : inférence ≤ 100 Mo de RAM, entraînement ≤ 1 Go, durée « pas des mois », pas d'énorme puissance de calcul.

> Tous les chiffres marqués **[calc]** sont produits par `tools/llm_budget.py` et reproductibles.
> Les chiffres marqués **[estim.]** sont des estimations d'ordre de grandeur, à valider par mesure.
> Les autres chiffres sont sourcés en §11.

---

## 0. Résumé exécutif

**Le verdict, sans détour.**

1. **100 Mo pour un LLM généraliste de qualité conversationnelle : non.** Le plancher réaliste d'un
   généraliste « utilisable » se situe autour de **300–500 Mo** (SmolLM2-360M en Q4, Qwen3-0.6B en Q4).
   SmolLM2-135M, le plus petit modèle réellement polyvalent publié, pèse **110 Mo en Q4_K_M pour ses seuls
   poids**, et **125 Mo de RSS** une fois le KV cache et le runtime comptés **[calc]**. Il est déjà à peine
   capable (MMLU 29,3 ; GSM8K 1,4 ; IFEval 29,9).
2. **100 Mo pour un modèle spécialisé, distillé et outillé : oui.** La cible crédible est
   **60–85 M paramètres en 4-bit QAT**, ou **~200 M paramètres en ternaire 1,58 bit**, avec un contexte de
   1 024–2 048 tokens et un KV cache quantifié. Budget type : 41 Mo de poids + 10 Mo de KV + 26 Mo de
   runtime = **78 Mo** **[calc]**.
3. **Le levier principal n'est pas la taille du modèle.** À budget mémoire constant, les gains les plus
   rentables sont, dans l'ordre : la **qualité et la distillation des données** (+10 à +20 points de
   benchmark à taille égale), le **vocabulaire** (jusqu'à 40 % du budget gaspillé par un tokenizer
   multilingue), le **KV cache** (facteur 4 entre f16 et q4), puis la quantification des poids.
4. **« Ne consomme rien » est un problème de temps, pas de joules par poids.** À 100 Mo, l'énergie
   incrémentale du modèle est de l'ordre de **3,5 mJ/token** ; l'énergie réellement payée au mur est
   `P_plateforme / débit`, soit **1,5 J/token à 10 tok/s sur une plateforme de 15 W** — 400 fois plus
   **[calc]**. Un modèle plus petit ne réduit la facture que s'il **augmente le débit** ou permet une
   plateforme moins puissante.
5. **Entraînement ≤ 1 Go : oui, sur un seul GPU, en jours.** 60–100 M paramètres avec Adam 8-bit,
   checkpointing et séquences courtes tiennent sous 1 Go. En revanche, **sur CPU c'est un non** :
   ~1,2 × 10¹⁸ FLOPs pour 100 M × 2 Md tokens, soit des centaines de jours à 50 GFLOPS utiles.
   Sur un GPU loué à l'heure, le même run coûte **quelques dollars** et prend **quelques heures**.

**Recommandation stratégique.** Ne pas chercher à faire « un petit ChatGPT ». Construire un
**modèle de 60–85 M paramètres, distillé depuis un grand modèle, spécialisé sur un domaine étroit,
augmenté par retrieval et contraint par grammaire**, et mesurer trois choses à chaque étape :
RSS réel, joules par réponse utile, et taux de réussite sur la tâche. C'est le seul chemin qui rend
« puissant » et « 100 Mo » compatibles.

**Ce que le projet ne pourra pas faire.** Raisonnement multi-étapes général, connaissances
encyclopédiques, code non trivial, mathématiques scolaires, multilinguisme large. Ces capacités
demandent de la capacité paramétrique, pas de l'ingénierie.

---

## 1. Cadrage : ce qu'on optimise exactement

### 1.1 Les trois contraintes, traduites en unités mesurables

| Contrainte énoncée | Traduction opérationnelle | Piège à éviter |
|---|---|---|
| « ~100 Mo de RAM » | RSS du processus au pic, poids + KV + runtime + scratch, hors OS | Le fichier `.gguf` n'est pas le RSS. Le KV cache peut dépasser les poids. |
| « Entraînement dans max 1 Go » | Mémoire de l'optimiseur + gradients + activations + batch | « 1 Go de données » est une autre contrainte (≈ 250 M tokens), très différente. |
| « Ne prend pas des mois » | Temps mural sur le matériel disponible | Le CPU est 50–200× plus lent qu'un GPU récent pour l'entraînement. |
| « Ne consomme rien » | Joules par réponse **utile**, pas par token | Un modèle lent consomme plus, même s'il est petit. |

**Ambiguïté n° 1 à lever (bloquante pour la suite) :** « 1 Go » désigne-t-il
(a) la mémoire du run d'entraînement, (b) le volume de données, ou (c) la taille du checkpoint ?
Ce rapport traite (a) comme hypothèse principale et documente (b) et (c) en §5.5.
Les trois mènent à des architectures différentes.

**Ambiguïté n° 2 :** « ne consomme rien » = énergie d'inférence, empreinte carbone, ou coût
matériel ? Les trois se rejoignent à petite échelle, mais pas au même endroit (§8).

### 1.2 Définition de « puissant »

Un modèle de 100 Mo ne sera jamais fort au sens général. On propose une définition opérationnelle :

> **Puissant = taux de réussite élevé sur un périmètre étroit, à coût marginal quasi nul.**

Trois axes à mesurer, dans cet ordre :
1. **Tâche** : exactitude sur un jeu de test du domaine (pas MMLU).
2. **Coût** : J/réponse juste, et non J/token.
3. **Encombrement** : RSS au 95ᵉ centile, pas la moyenne.

Un modèle qui répond juste 70 % du temps en 0,4 J est plus « puissant » qu'un modèle qui répond
juste 45 % du temps en 3 J. C'est ce rapport qu'il faut optimiser.

---

## 2. La contrainte reine : l'arithmétique des 100 Mo

### 2.1 Décomposition

```
RSS = poids + KV cache + runtime + activations/scratch
```

avec :

```
poids        = params × bits_par_poids / 8
KV/token     = 2 × couches × têtes_KV × (d_model / têtes) × octets_par_élément
KV total     = KV/token × contexte
```

Le terme `2` du KV correspond à K et V. `têtes_KV` est le nombre de têtes KV (GQA/MQA), pas le
nombre de têtes d'attention.

### 2.2 Ce que ça donne concrètement

| Modèle | Params | Poids | KV (2k, q8) | Runtime+scratch | **RSS** | < 100 Mo ? |
|---|---:|---:|---:|---:|---:|---|
| SmolLM2-135M, Q4.5 | 135 M | 75,7 Mo | 23,6 Mo | 26 Mo | **125 Mo** | non |
| SmolLM2-135M, Q2.6 | 135 M | 43,7 Mo | 23,6 Mo | 26 Mo | **93 Mo** | oui, mais qualité dégradée |
| MobileLLM-R1-140M, Q4.5 | 140 M | 78,8 Mo | 11,8 Mo | 26 Mo | **117 Mo** | non |
| Gemma 3 270M, Q4.5 | 261 M | 146,6 Mo | 23,6 Mo | 26 Mo | **196 Mo** | non |
| **NanoFR-60M, Q4.5** | **59 M** | **33,4 Mo** | **10,5 Mo** | **26 Mo** | **70 Mo** | **oui** |
| **NanoFR-75M, Q4.5** | **73 M** | **41,2 Mo** | **10,5 Mo** | **26 Mo** | **78 Mo** | **oui** |
| **NanoFR-85M, Q4.5** | **84 M** | **47,5 Mo** | **12,6 Mo** | **26 Mo** | **86 Mo** | **oui** |
| NanoFR-125M, Q4.5 | 125 M | 70,4 Mo | 23,6 Mo | 26 Mo | **120 Mo** | non |
| **Ternaire-200M, 1.58 bit** | **201 M** | **39,8 Mo** | **29,4 Mo** | **26 Mo** | **95 Mo** | **oui (juste)** |

*[calc] — KV en q8_0 (1 octet/élément), contexte 2 048, runtime supposé 20 Mo + 6 Mo de scratch.*

**Trois enseignements immédiats.**

1. **À 4,5 bits/poids, la frontière est autour de 85 M paramètres** pour 100 Mo avec 2 k de contexte.
   Au-delà, il faut soit un contexte plus court, soit un KV plus agressif, soit un runtime plus léger.
2. **Le KV cache n'est pas un détail** : 23,6 Mo pour SmolLM2-135M, soit 19 % du RSS. À contexte 8 k,
   il devient 94 Mo et fait exploser n'importe quel budget de 100 Mo. **Le contexte est un paramètre
   de premier ordre, pas une option.**
3. **Le ternaire change l'arithmétique** : 201 M paramètres tiennent dans 40 Mo de poids. C'est
   l'équivalent, en mémoire, d'un modèle 4-bit de ~70 M paramètres — mais avec 3× plus de capacité.

### 2.3 Le piège du vocabulaire

C'est le poste le plus sous-estimé d'un petit modèle.

```
params_embedding = vocabulaire × d_model
```

À d_model = 576 :
- vocab 32 768 → **18,9 M paramètres** (≈ 10 Mo en 4-bit)
- vocab 128 000 → **73,7 M paramètres** (≈ 41 Mo en 4-bit)
- vocab 262 144 → **151 M paramètres** (≈ 85 Mo en 4-bit)

Gemma 3 270M consacre **170 M de ses 270 M de paramètres à sa table d'embeddings** (vocab 256 k,
contexte 32 k). MobileLLM-R1-140M utilise un vocabulaire de 128 k : plus de la moitié du modèle est
dans les embeddings. C'est un choix défendable pour un modèle multilingue généraliste ; c'est
**ruineux** pour un modèle francophone de 100 Mo.

À l'inverse, la littérature sur les tokenizers montre que **l'efficacité de tokenisation plafonne
autour de 32 k** de vocabulaire : au-delà, on paie des paramètres sans gagner proportionnellement en
compression de texte. Pour le français, un vocabulaire de **32 k** (ou 24 k si l'on veut gagner 6 Mo)
est le bon point de fonctionnement.

**Décision proposée : vocabulaire 32 768, embeddings partagés entrée/sortie (tied).**
Gain immédiat : ~30 Mo de budget, soit 30 % de la contrainte.

### 2.4 Le piège du contexte et du KV

Le KV cache croît linéairement avec le contexte, et c'est lui qui décide si le modèle « tient » :

| Contexte | KV f16 | KV q8 | KV q4 |
|---:|---:|---:|---:|
| 512 | 5,9 Mo | 2,9 Mo | 1,5 Mo |
| 1 024 | 11,8 Mo | 5,9 Mo | 2,9 Mo |
| 2 048 | 23,6 Mo | 11,8 Mo | 5,9 Mo |
| 4 096 | 47,2 Mo | 23,6 Mo | 11,8 Mo |
| 8 192 | 94,4 Mo | 47,2 Mo | 23,6 Mo |

*[calc] pour MobileLLM-R1-140M : 15 couches, 3 têtes KV, head_dim 64.*

Quatre leviers, du moins risqué au plus radical :
1. **GQA/MQA** : passer de 9 têtes KV à 1 divise le KV par 9. Coût qualité faible à cette taille.
2. **KV quantifié q8** : division par 2, perte quasi nulle (0,0 à +0,1 % de perplexité mesurée).
3. **KV q4** : division par 4, +0,5 à +1,1 % de perplexité, mais décodage plus lent sur certains
   backends (déquantification à chaque pas). À réserver aux contextes longs.
4. **Fenêtre glissante / architecture récurrente** : le KV devient borné, indépendant du contexte.
   C'est la seule solution qui permette un contexte long sous 100 Mo (§4.6).

**Recommandation : contexte 2 048, KV q8, 2 à 3 têtes KV.** Si le besoin produit est un contexte plus
long, il faut changer d'architecture, pas comprimer davantage.

### 2.5 Ce que 100 Mo n'achètent pas

| Capacité | Faisable à 100 Mo ? | Commentaire |
|---|---|---|
| Répondre à des questions factuelles fermées | Oui, avec RAG | La connaissance est dans l'index, pas dans les poids. |
| Suivre une instruction courte, formatée | Oui | SFT + grammaire contrainte. |
| Résumer un texte de 2 000 tokens | Partiellement | Contexte limite ; qualité variable. |
| Converser de façon générale | Non | SmolLM2-135M échoue déjà (MT-Bench 1,98/10). |
| Raisonner en plusieurs étapes | Non | GSM8K 1,4 à 135 M ; ~16 à 140 M spécialisé. |
| Écrire du code non trivial | Non | HumanEval ≈ 0–16 % selon spécialisation. |
| Être multilingue large | Non | Le vocabulaire coûterait plus que le modèle. |
| Apprendre de nouveaux faits après déploiement | Non (poids) / Oui (RAG) | Séparer mémoire paramétrique et non paramétrique. |

---

## 3. État de l'art : qui tient vraiment dans 100 Mo ?

### 3.1 Le tableau de référence

| Modèle | Params | Tokens d'entraînement | Poids Q4 | Forces | Limites |
|---|---:|---:|---:|---|---|
| **TinyStories** (2023) | 1–10 M | ~1 Md | 1–5 Mo | Texte cohérent, grammaire quasi parfaite sur un domaine jouet | Domaine volontairement simplifié ; ne généralise pas |
| **SmolLM2-135M** (2024) | 135 M | 2 000 Md | ~110 Mo | Petit généraliste solide, très documenté | MMLU 29,3 ; GSM8K 1,4 ; déjà hors budget |
| **MobileLLM-125M** (2024) | 125 M | 1 000 Md | ~80 Mo | Deep-thin + GQA : +2,7 pts vs OPT-125M | Toujours ~125 Mo de RSS |
| **MobileLLM-R1-140M** (2025) | 140 M | 4 200 Md | ~80 Mo | **GSM8K 16,3 ; HumanEval 15,9** — raisonnement réel à 140 M | Vocab 128 k → embeddings énormes ; hors budget |
| **Gemma 3 270M** (2025) | 270 M | 6 000 Md | ~125 Mo (INT4 QAT) | Très économe en batterie, 32 k de contexte | 170 M de paramètres dans le vocabulaire ; hors budget |
| **Qwen3-0.6B** (2025) | 600 M | 36 000 Md | ~350 Mo | Généraliste fort pour sa taille | 3,5× le budget |
| **BitNet b1.58 2B4T** (2025) | 2 400 M | 4 000 Md | ~400 Mo (1,58 bit) | 0,028 J/token, 29 ms/token CPU, qualité Qwen2.5-1.5B | 4× le budget ; peu de modèles ternaires < 500 M |
| **LFM2-350M / Granite-350M** | 350 M | — | ~200 Mo | Architectures modernes efficaces | Hors budget |

### 3.2 Ce que l'état de l'art prouve

1. **La capacité émerge plus tôt qu'on ne le croyait, mais seulement sur des distributions étroites.**
   TinyStories : 10 M de paramètres suffisent pour une grammaire quasi parfaite *si le domaine est
   simple*. SmolLM2-135M : 2 000 Md de tokens ne suffisent pas pour le raisonnement général.
2. **L'architecture compte autant que la taille à petite échelle.** MobileLLM montre que
   « profond et étroit » (30 couches, d=576) bat « large et court » (12 couches, d=768) à paramètres
   égaux, avec +2,7 points. L'embedding partagé économise 11,8 % des paramètres pour −0,2 point.
3. **Les données comptent plus que les paramètres.** MobileLLM-R1-140M atteint 16,3 % sur GSM8K
   avec 4,2 T tokens, là où SmolLM2-135M plafonne à 1,8 % avec 2 T tokens. Même ordre de taille,
   qualité de données et distillation différentes.
4. **Le ternaire natif est réel mais immature sous 1 Md.** BitNet b1.58 2B4T prouve le concept à
   2 Md ; il n'existe pas encore de modèle ternaire de 100–300 M entraîné sérieusement et publié.
   C'est un pari de recherche, pas une valeur sûre.
5. **Personne ne publie de LLM généraliste sous 100 Mo.** Le plus petit généraliste crédible est à
   ~110 Mo de poids seuls. **Il n'y a donc pas de solution toute faite : il faut construire.**

### 3.3 Leçons directement réutilisables

- **Deep-thin + GQA + SwiGLU + tied embeddings** : socle gratuit, +3 à +4 points à taille égale.
- **Vocabulaire ≤ 32 k** : indispensable sous 100 Mo.
- **Distillation depuis un modèle 2–10× plus gros** : le meilleur ratio qualité/coût connu
  (MiniLLM : 120 M distillé de 1,5 B gagne +6,1 points de score GPT-4 vs SFT).
- **SFT court sur données ciblées** : c'est là que se gagne l'utilité perçue.
- **QAT plutôt que PTQ sous 4 bits** : la quantification post-entraînement s'effondre sous 3 bits,
  surtout sur les petits modèles.

---

## 4. Les leviers, classés par retour sur investissement

### 4.1 Quantification : le sweet spot est 4 bits, pas 2

| Format | Bits/poids | Perte de perplexité | Verdict à 100 Mo |
|---|---:|---|---|
| FP16 | 16 | référence | Non : 2× le budget |
| Q8 | ~8 | ~0 | Non : pas assez de compression |
| **Q4_K_M / QAT 4-bit** | **4,5** | **+3 à +5 %** | **Oui : le défaut** |
| Q3 | ~3,3 | +8 à +15 % | Seulement si contrainte forte |
| Q2 | ~2,6 | +15 à +25 %, s'effondre sur les petits modèles | Non, sauf modèle ternaire natif |
| **Ternaire natif (BitNet)** | **1,58** | dépend de l'entraînement | **Pari intéressant** |

**Point clé : la quantification post-entraînement (PTQ) est un mauvais deal sous 4 bits.** Les petits
modèles ont peu de redondance ; à 2 bits, la perplexité peut augmenter de deux ordres de grandeur.
Deux options sérieuses :
- **QAT 4-bit** : on simule la quantification pendant l'entraînement. Google l'utilise pour Gemma 3
  (INT4 QAT, perte < 2 %). C'est la voie standard pour tenir 4 bits sans s'effondrer.
- **Ternaire natif** : on entraîne directement avec des poids {−1, 0, +1} et des activations int8.
  ~1,58 bit/poids, et les multiplications deviennent des additions. Le gain énergétique est réel
  (−55 à −82 % selon la plateforme), mais il faut entraîner depuis zéro.

**Recommandation : QAT 4-bit pour la piste sûre ; ternaire pour la piste exploratoire.**

### 4.2 Architecture : ce qui est gratuit et ce qui ne l'est pas

| Choix | Gain | Coût | Recommandation |
|---|---|---|---|
| Deep-thin (24–30 couches, d=448–576) | +2 à +3 pts | Plus lent en parallèle, plus de latence par couche | **Oui** |
| GQA (2–3 têtes KV) | KV ÷ 3 à ÷ 9 | −0,2 pt | **Oui, obligatoire** |
| Tied embeddings | −10 à −15 % de paramètres | −0,2 à −0,6 pt | **Oui** |
| SwiGLU | +1,3 pt | +33 % de paramètres FFN | **Oui** |
| Vocab 32 k | −30 Mo | perte de compression sur le code/rare | **Oui** |
| Layer sharing | +0,7 pt à taille égale | latence +2,6 % ; moins de diversité | À tester (E3) |
| MoE | +capacité à FLOPs constants | **+mémoire** (tous les experts sont en RAM) | **Non** : le décodage CPU est memory-bound |
| Sparsité non structurée | 0 en pratique | Aucun kernel CPU rapide | **Non** |
| Élagage structuré | −20 à −30 % de poids | −2 à −5 pts, à re-entraîner | Plus tard, si besoin |

**Pourquoi le MoE est un piège ici :** en décodage batch=1, on lit tous les poids actifs depuis la
DRAM. Un MoE réduit les FLOPs mais **augmente** l'empreinte mémoire, et le débit est limité par la
bande passante, pas par le calcul. À 100 Mo, c'est exactement l'inverse de ce qu'on veut.

### 4.3 Distillation : le meilleur levier qualité par euro

**Principe.** Un enseignant plus gros (2 à 10× le student) fournit des distributions de sortie, pas
seulement des réponses. Le student apprend plus vite, avec moins de tokens.

**Preuves.**
- MiniLLM : GPT-2 120 M passe de 38,6 (SFT) à 44,7 (distillé de 1,5 B) sur DollyEval.
- MobileLLM-R1 utilise Llama-3.1-8B-Instruct comme enseignant en mid-training, sur 200 Md tokens.
- Les revues de distillation confirment qu'au-delà d'un ratio 10:1 enseignant/étudiant, le gain
  s'effondre (problème de « capacité gap »).

**Recette recommandée.**
1. Enseignant : un modèle 1–8 B, idéalement déjà bon en français et sur le domaine.
2. Générer 50 k à 500 k exemples de qualité (pas 2 T tokens web).
3. Distiller en deux temps : pré-entraînement sur texte (logits top-k) puis SFT on-policy (MiniLLM).
4. Garder 5–10 % de données réelles pour éviter l'effondrement de diversité.

**Coût.** 500 k exemples × 500 tokens = 250 M tokens de génération enseignant. C'est le poste
dominant, pas l'entraînement du student.

### 4.4 Données : la qualité avant le volume

- **TinyStories** : 10 M de paramètres suffisent pour une grammaire quasi parfaite sur un corpus
  simplifié. La leçon n'est pas « les petits modèles sont doués », c'est **« un petit modèle ne peut
  apprendre que ce que sa distribution contient »**.
- **SmolLM2** : ablations systématiques, 60 % FineWeb-Edu / 40 % DCLM, ajout progressif de code et de
  maths, décroissance du LR sur les 10 % finaux. 2 T tokens pour 135 M.
- **MobileLLM-R1** : 2 T tokens haute qualité + distillation suffisent pour battre des modèles
  entraînés sur 36 T.

**Conclusion opérationnelle.** Pour 100 Mo, il ne faut **pas** viser la couverture mondiale. Il faut
un corpus de 1 à 10 Md tokens, très filtré, centré sur le domaine, et **beaucoup** d'exemples
d'instruction synthétiques générés par l'enseignant.

### 4.5 Le système autour du modèle : le multiplicateur oublié

| Technique | Gain mesuré | Coût mémoire | Coin |
|---|---|---|---|
| **RAG (BM25/embedding)** | Jusqu'à +49,2 % d'exactitude sur les petits modèles ; Flan-T5-small atteint 78,1 % avec passages oracle | Index hors modèle (disque/SSD) | **Le meilleur levier absolu** |
| **Outils / function calling** | Un agent 4 B outillé bat un 32 B sans outils sur GAIA (18,18 % vs 12,73 %) | Nul | Très rentable |
| **Décodage contraint par grammaire** | Supprime les sorties invalides (JSON, dates, IDs) | ~0 | **Obligatoire** en production |
| **Cascade** (petit modèle → grand modèle si confiance faible) | 70–90 % des requêtes traitées à coût nul | Nul côté petit | À concevoir dès le départ |
| **Mémoire externe** (faits, préférences) | Évite de réentraîner | Faible | Oui |
| **Spéculatif** (draft model) | 2–3× de débit | +un petit modèle | Utile seulement si > 1 tok/s |

**Le point stratégique :** sous 100 Mo, on ne peut pas mettre le monde dans les poids. On met la
**connaissance dans l'index**, la **compétence dans les poids**, et la **rigueur dans la grammaire**.
C'est l'architecture d'un produit, pas d'un benchmark.

### 4.6 Architectures alternatives : RWKV, Mamba, xLSTM

Ces architectures suppriment le KV cache (état fixe). À contexte long, l'avantage est massif :
Mamba est 12,46× plus économe en mémoire et 10,67× plus rapide qu'un Transformer à 4 096 tokens.
xLSTM 7B génère ~50 % plus vite que Mamba et à mémoire constante.

**Mais** :
- les implémentations CPU grand public sont rares et jeunes ;
- les poids pré-entraînés < 300 M sont quasi inexistants ;
- l'écosystème (llama.cpp, tokenizers, quantization) est centré sur le Transformer.

**Verdict.** Pour un contexte ≤ 2 k, un Transformer GQA est plus simple, mieux outillé et aussi
économe. Pour un contexte ≥ 8 k sous 100 Mo, **une architecture hybride** (majorité de couches
récurrentes + quelques couches d'attention) devient la seule option crédible. À garder en piste B,
pas en piste principale.

### 4.7 Décodage : où se joue la latence perçue

- **Greedy/beam** : déterministe, bon pour les tâches fermées.
- **Décodage contraint** : indispensable pour les formats.
- **Early exit / couches sautées** :gain de 10–30 % sur les tokens faciles, mais complexité.
- **Batching** : n'a d'intérêt que côté serveur ; en local, batch=1.

### 4.8 Tableau de synthèse ROI

| Levier | Gain qualité | Gain mémoire | Effort | Priorité |
|---|---|---|---|---|
| Données filtrées + distillation | ★★★★★ | — | Moyen | **P0** |
| Vocab 32 k + tied embeddings | ★★ | ★★★★★ (−30 Mo) | Faible | **P0** |
| GQA + KV q8 | ★ | ★★★★ (−12 à −20 Mo) | Faible | **P0** |
| QAT 4-bit | ★★★ | ★★★★★ | Moyen | **P0** |
| RAG + outils | ★★★★★ (sur la tâche) | 0 | Moyen | **P0** |
| Deep-thin + SwiGLU | ★★★ | — | Faible | **P1** |
| Contexte 2 k au lieu de 8 k | ★ | ★★★★★ | Faible | **P1** |
| Ternaire natif | ★★ (incertain) | ★★★★★ | Élevé | **P2** |
| Architecture récurrente | ★★ | ★★★★ (à long contexte) | Très élevé | **P3** |
| MoE / sparsité | ★ | ★ (négatif) | Élevé | **Non** |

---

## 5. Entraîner dans 1 Go, sans y passer des mois

### 5.1 L'arithmétique mémoire de l'entraînement

Pour un entraînement Adam classique en mixed precision :

| Poste | Octets/paramètre | 60 M params | 100 M params | 200 M params |
|---|---:|---:|---:|---:|
| Poids maîtres fp32 | 4 | 240 Mo | 400 Mo | 800 Mo |
| Gradients fp32 | 4 | 240 Mo | 400 Mo | 800 Mo |
| Adam m + v fp32 | 8 | 480 Mo | 800 Mo | 1 600 Mo |
| **Total fp32** | **16** | **960 Mo** | **1 600 Mo** | **3 200 Mo** |
| **Adam 8-bit** | **10** | **600 Mo** | **1 000 Mo** | **2 000 Mo** |
| Adam bf16 (instable) | 6 | 360 Mo | 600 Mo | 1 200 Mo |

*Sans activations ni batch. Le total réel est plus élevé.*

**Conséquences.**
- **100 M paramètres en Adam fp32 ne tient pas dans 1 Go.** Il faut Adam 8-bit (~1,0 Go) et des
  séquences courtes, ou descendre à 60–80 M.
- **Les activations sont le poste caché.** Avec un vocabulaire de 32 k et un batch de 512 tokens,
  la matrice de logits seule fait `512 × 32 000 × 4 octets = 65 Mo`. Le gradient checkpointing et le
  calcul de perte par chunks sont obligatoires.
- **LoRA/QLoRA** ne réduit pas les poids de base : il réduit les gradients et les états d'optimiseur.
  Pour un modèle de 60 M, le gain est marginal ; pour affiner un modèle existant de 300 M+, c'est
  la seule voie sous 1 Go.

### 5.2 L'arithmétique compute

```
FLOPs ≈ 6 × N × D          (D = tokens, N = paramètres non-embedding)
```

Pour 60–100 M paramètres et 2 Md tokens (Chinchilla : 20 tokens/paramètre) :

| Modèle | Tokens | FLOPs | CPU 50 GFLOPS | 1 GPU 100 TFLOPS @30 % MFU |
|---|---:|---:|---:|---:|
| 60 M | 1,2 Md | 4,3 × 10¹⁷ | ~100 jours | ~1,7 h |
| 100 M | 2,0 Md | 1,2 × 10¹⁸ | ~280 jours | ~4,6 h |
| 200 M | 4,0 Md | 4,8 × 10¹⁸ | ~1 100 jours | ~18 h |

*[calc, ordres de grandeur]*

**Le message est net : l'entraînement CPU n'est pas viable. Un seul GPU, même modeste, change tout.**
Un run 100 M / 2 Md tokens représente ~8 à 15 GPU-heures ; loué à 0,5–1 $/h, c'est **moins de 15 $**.
Il existe même des récits reproductibles de modèles 97,5 M entraînés sur 2 Md tokens en **2 h 23 sur
une RTX 5090 pour 1,40 $**.

### 5.3 Les trois scénarios matériels

| Scénario | Faisable | Durée | Ce qu'on obtient |
|---|---|---|---|
| **CPU seul, 16–32 cœurs, 1 Go** | Oui pour 20–40 M params, non pour 100 M | Semaines à mois | Preuve de concept, qualité faible |
| **1 GPU 8–24 Go (local ou loué)** | Oui, confortable | Heures à jours | **La cible recommandée** |
| **1 GPU + 1 Go de VRAM seulement** | 60–80 M avec Adam 8-bit, seq 1 024 | Jours | Faisable, contraint |
| **Plusieurs GPU** | Oui, mais hors sujet | Heures | Inutile à cette échelle |

**Si le « 1 Go » est une contrainte de VRAM stricte** : viser **60–70 M paramètres**, séquences de
1 024, batch 8–16, Adam 8-bit, gradient checkpointing, activation recomputation sur les blocs FFN.
C'est un run de quelques jours sur une carte 8 Go, pas de quelques mois.

### 5.4 Recette d'entraînement recommandée (5 étapes)

1. **Tokeniser** un corpus français filtré + le domaine (1–5 Md tokens), vocab 32 k.
2. **Pré-entraîner** 60–85 M paramètres sur 20–40 Md tokens ? Non : commencer par 1–2 Md tokens
   pour valider la courbe. 20 tokens/paramètre est un plancher, pas un objectif.
3. **Mid-trainer** avec distillation d'un enseignant 1–8 B sur 100–300 M tokens.
4. **SFT** sur 50–200 k instructions du domaine, format court.
5. **QAT 4-bit** + export GGUF/llama.cpp, puis **mesure** du RSS et des J/token.

**Durée totale réaliste : 1 à 3 semaines**, dont l'essentiel en préparation de données et en
évaluation. L'entraînement lui-même est l'étape la plus courte.

### 5.5 Les autres lectures de « 1 Go »

- **1 Go de données** ≈ 250 M tokens (à ~4 octets/token). C'est **très peu** : suffisant pour du
  fine-tuning spécialisé ou pour TinyStories, insuffisant pour un généraliste. Dans ce cas, la
  stratégie devient : partir d'un modèle existant (SmolLM2-135M) et le **distiller/affiner** sur
  250 M tokens de domaine, puis le quantifier.
- **1 Go de checkpoint** : 4 bits × 100 M ≈ 50 Mo, donc 1 Go n'est pas contraignant à cette taille.
  Cette lecture est peu probable.

---

## 6. Architecture candidate

### 6.1 Piste A (recommandée) — NanoFR-75M, 4-bit QAT

| Paramètre | Valeur | Justification |
|---|---|---|
| Couches | 20 | Deep-thin ; 30 couches coûte trop de KV sous 100 Mo |
| d_model | 512 | Compromis capacité/latence |
| Têtes / têtes KV | 8 / 2 | GQA facteur 4 sur le KV |
| FFN (SwiGLU) | 1 408 | Ratio ~2,75× |
| Vocabulaire | 32 768 | Efficacité de tokenisation plateau + budget |
| Embeddings | partagés | −10 % de paramètres |
| Contexte | 2 048 | KV 10,5 Mo en q8 |
| Quantification | QAT 4-bit (4,5 bits effectifs) | Perte < 2 % visée |
| **Params** | **~73 M** | |
| **Poids** | **~41 Mo** | |
| **KV (2 k, q8)** | **~10,5 Mo** | |
| **RSS estimé** | **~78 Mo** | Marge de 22 Mo |
| Énergie (borne basse) | ~3,5 mJ/token | |
| FLOPs/token | ~0,15 GFLOP | |

### 6.2 Piste B (exploratoire) — Ternaire-200M

| Paramètre | Valeur |
|---|---|
| Couches | 28 |
| d_model | 768 |
| Têtes / têtes KV | 12 / 4 |
| FFN | 2 048 |
| Vocabulaire | 32 768, partagé |
| Contexte | 2 048 (KV 29,4 Mo en q8) |
| Quantification | Ternaire natif 1,58 bit, activations int8 |
| **Params** | **~201 M** |
| **Poids** | **~40 Mo** |
| **RSS estimé** | **~95 Mo** (juste) |
| Avantage | 3× plus de capacité pour le même poids |
| Risque | Aucun modèle ternaire < 300 M validé ; outillage immature |

**Verdict :** la piste B est plus ambitieuse et probablement plus performante à budget égal, mais
elle ajoute un risque d'exécution majeur (kernels, qualité, écosystème). **Commencer par A, garder B
comme expérience parallèle E6.**

### 6.3 Piste C — « ne pas entraîner » : distiller SmolLM2-135M

Partir de SmolLM2-135M, remplacer son vocabulaire par un vocab 32 k (ou garder 49 k), le
spécialiser par distillation + LoRA, puis le quantifier en QAT. Avantage : pas de pré-entraînement,
résultat en jours. Inconvénient : on hérite de son plafond (MMLU 29,3) et on ne tient pas dans
100 Mo sans réduire le contexte à 1 k ou passer en Q2. **C'est le meilleur plan B si le temps manque.**

### 6.4 Ce qu'il faut mesurer avant de coder

1. Le RSS réel de llama.cpp avec un modèle de 41 Mo (hypothèse : 20–30 Mo de runtime).
2. Le débit CPU d'un 75 M en Q4 sur la machine cible.
3. La qualité d'un 75 M distillé vs un 135 M quantifié : à quelle taille le petit modèle devient-il
   inutile ?
4. Le coût réel de génération des données de distillation.

---

## 7. Alternatives non-LLM : quand « rien » est la bonne réponse

Il faut être honnête : pour beaucoup de cas d'usage, un LLM de 100 Mo est le mauvais outil.

| Besoin | Meilleure solution | Coût | Pourquoi pas un LLM |
|---|---|---|---|
| Classification / routage | Régression logistique, petit BERT | < 5 Mo | Plus rapide, plus précis, explicable |
| Extraction de champs | Regex + règles + petit modèle | < 50 Mo | Déterministe, auditable |
| FAQ fermée | BM25 / embeddings + réponse template | Index seul | Pas d'hallucination |
| Calcul, dates, unités | Code / outils | 0 | Exact |
| Rédaction contrainte | Templates + grammaire | 0 | Contrôlable |
| Dialogue ouvert | LLM 300 Mo+ | 300 Mo | 100 Mo ne suffit pas |

**Architecture hybride recommandée** : un routeur (règles ou petit classifieur) envoie 70–90 % des
requêtes vers des chemins déterministes, et le LLM ne traite que la frange qui demande du langage.
C'est ainsi qu'on obtient un système « qui ne consomme rien » : en ne l'allumant pas.

---

## 8. Métrologie : mesurer pour ne pas se mentir

### 8.1 Mémoire

- **RSS** (`/proc/<pid>/status`, `smaps_rollup`) : ce que le système facture. Attention aux pages
  mmap partagées et au page cache.
- **PSS** : plus juste pour un processus qui mappe des poids.
- Mesurer **au pic**, après 3 générations complètes, contexte rempli.
- Toujours rapporter : modèle, quantification, contexte, backend, version.

### 8.2 Énergie

Trois niveaux, du moins cher au plus fiable :
1. **RAPL** (Intel/AMD, Linux) : énergie CPU + DRAM, précision correcte, gratuit.
2. **Joulescope / INA219** : mesure au rail d'alimentation, précision excellente, ~100–500 $.
3. **Prise connectée** : énergie murale, inclut l'écran et l'OS, pertinent pour l'usage réel.

**Toujours rapporter** :
- **J/token** et **J/réponse juste** ;
- le **débit** (tok/s) au moment de la mesure ;
- la **puissance de la plateforme** (car `E ≈ P / débit`).

**Le piège du « race-to-idle ».** Sur un CPU à puissance quasi constante, l'énergie par token est
proportionnelle au temps d'exécution. Un modèle plus petit mais mal optimisé peut consommer
**davantage** qu'un modèle plus gros et mieux vectorisé. Des mesures sur Intel Core Ultra 7 montrent
que les cœurs E coûtent 2,3× les joules par token des cœurs P — à cause de la durée, pas de la
puissance. **Optimiser le débit est souvent plus efficace que réduire la taille.**

### 8.3 Qualité

- **Tâche cible** : jeu de test tenu à l'écart, 500–2 000 exemples, métrique binaire.
- **Général** : MMLU (connaissances), HellaSwag/ARC (bon sens), GSM8K (maths), IFEval (instructions).
- **Français** : FLUE, FrenchBench, ou un jeu maison.
- **Robustesse** : sorties hors format, prompts adverses, entrées longues.
- **Ne pas comparer** un modèle distillé sur le domaine à un généraliste sur MMLU : ce n'est pas
  la même question.

---

## 9. Plan d'expériences

| # | Expérience | Question | Critère go/no-go | Effort |
|---|---|---|---|---|
| **E0** | Mesure du plancher mémoire | Quel est le RSS d'un modèle 40 Mo dans llama.cpp ? | < 30 Mo de runtime → budget viable | 1 j |
| **E1** | NanoFR-60M from scratch, 1 Md tokens | Un 60 M apprend-il quelque chose d'utile ? | Perplexité < 25 sur le domaine ; génération grammaticale | 3 j |
| **E2** | Vocab 16 k vs 32 k vs 49 k | Quel est le vrai coût du vocabulaire ? | 32 k retenu si gain de compression > perte | 2 j |
| **E3** | Deep-thin vs wide, layer sharing | Confirme-t-on MobileLLM à 60 M ? | +1,5 pt pour deep-thin | 3 j |
| **E4** | Distillation 1,5 B → 75 M | Combien de points la distillation rapporte-t-elle ? | +5 pts vs SFT seul | 5 j |
| **E5** | QAT 4-bit vs PTQ 4-bit | La QAT tient-elle la qualité ? | < 2 % de perte | 3 j |
| **E6** | Ternaire 200 M | Le pari ternaire est-il viable sous 300 M ? | Qualité ≥ NanoFR-75M à budget égal | 2 sem. |
| **E7** | RAG + grammaire | Le système bat-il le modèle seul ? | +15 pts sur la tâche | 3 j |
| **E8** | Mesure énergie | J/réponse juste sur la cible | < 0,5 J/réponse | 1 j |

**Ordre recommandé : E0 → E1 → E4 → E5 → E7**, puis E2/E3 en optimisation, E6 en exploration.
Les expériences E0 et E7 sont les moins chères et les plus informatives.

---

## 10. Risques et angles morts

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| Le runtime dépasse 30 Mo | Moyenne | Budget explosé | Mesurer tôt (E0) ; envisager un moteur C maison |
| Le 75 M distillé reste inutile | Moyenne | Projet sans produit | Définir la tâche avant d'entraîner ; RAG |
| Les données françaises de qualité manquent | Élevée | Qualité plafonnée | Traduire/distiller ; corpus ciblé |
| Le ternaire ne passe pas à l'échelle 200 M | Élevée | Piste B morte | Garder la piste A comme référence |
| « 1 Go » mal interprété | Moyenne | Mauvaise architecture | Trancher la question (§1.1) |
| La mesure d'énergie est non reproductible | Élevée | Chiffres non comparables | Protocole §8 + matériel documenté |
| Le modèle est « bon » sur le test et mauvais en usage | Moyenne | Faux positif | Jeu de test externe, évaluation humaine |

**Angle mort majeur : la latence.** Un modèle de 75 M peut être plus lent qu'un 135 M si ses couches
sont mal vectorisées ou si le backend ne supporte pas ses noyaux. La latence doit être mesurée, pas
déduite du nombre de paramètres.

**Second angle mort : la qualité du tokenizer français.** Un vocabulaire sous-entraîné sur le
français produit des tokens plus nombreux, ce qui augmente le coût et réduit le contexte utile.
Il faut mesurer la fertilité (tokens/mot) sur un corpus français avant de figer le vocabulaire.

---

## 11. Sources

### Modèles et architectures
1. Eldan & Li, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*, arXiv:2305.07759 — https://arxiv.org/abs/2305.07759
2. Liu et al., *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases*, arXiv:2402.14905 — https://arxiv.org/abs/2402.14905
3. Meta, *MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes*, arXiv:2509.24945 — https://arxiv.org/abs/2509.24945
4. Allal et al., *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model*, arXiv:2502.02737 — https://arxiv.org/abs/2502.02737
5. Google, *Introducing Gemma 3 270M*, Google Developers Blog, août 2025 — https://developers.googleblog.com/en/introducing-gemma-3-270m/

### Quantification et efficacité
6. Ma et al., *BitNet b1.58 2B4T Technical Report*, arXiv:2504.12285 — https://arxiv.org/abs/2504.12285
7. Wang et al., *1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs*, arXiv:2410.16144 — https://arxiv.org/abs/2410.16144
8. Microsoft, *BitNet* (dépôt officiel) — https://github.com/microsoft/BitNet
9. Xu et al., *OneBit: Towards Extremely Low-bit Large Language Models*, NeurIPS 2024 — https://arxiv.org/abs/2402.11295
10. Google, *Gemma 3 QAT Models*, Google Developers Blog, avril 2025 — https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/

### Distillation
11. Gu et al., *MiniLLM: Knowledge Distillation of Large Language Models*, arXiv:2306.08543 — https://arxiv.org/abs/2306.08543
12. Xu et al., *On-Policy Distillation of Large Language Models* (MiniLLM, v5) — https://arxiv.org/html/2306.08543v5

### Entraînement et scaling
13. Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla), arXiv:2203.15556 — https://arxiv.org/abs/2203.15556
14. *Training a 97.5M Python coding model from scratch for $1.40 in GPU time*, DEV Community, 2026 — https://dev.to/tmgbedu/training-a-975m-python-coding-model-from-scratch-for-140-in-gpu-time-4lp4

### Systèmes et retrieval
15. Sarthi et al., *RAG in the Wild: On the (In)effectiveness of LLMs with Mixture-of-Knowledge Retrieval Augmentation*, arXiv:2507.20059 — https://arxiv.org/abs/2507.20059
16. *Retrieval Helps or Hurts? A Deeper Dive into the Efficacy of Retrieval Augmentation to Language Models*, arXiv:2402.13492 — https://arxiv.org/abs/2402.13492
17. Izacard et al., *Atlas: Few-shot Learning with Retrieval Augmented Language Models*, JMLR 2023 — http://www.jmlr.org/papers/volume24/23-0037/23-0037.pdf

### Énergie et métrologie
18. Horowitz, *Computing's Energy Problem (and what we can do about it)*, ISSCC 2014 — modèle énergétique 640 pJ / lecture DRAM 32 bits.
19. *Energy Consumption of Quantized LLM Inference on Raspberry Pi 4* (Joulescope), ACM, 2026 — https://dl.acm.org/doi/pdf/10.1145/3767742
20. *tokenwatt: Energy characterisation of LLM inference across heterogeneous CPU islands* — https://github.com/etisamhaq/tokenwatt
21. Chung et al., *The ML.ENERGY Benchmark*, arXiv:2505.06371 — https://arxiv.org/abs/2505.06371
22. *TokenPowerBench: Benchmarking the Power Consumption of LLM Inference*, arXiv:2512.03024 — https://arxiv.org/html/2512.03024v1

### Tokenizers et vocabulaire
23. *Efficient Vocabulary Reduction for Small Language Models*, COLING 2025 — https://aclanthology.org/2025.coling-industry.64.pdf
24. *To Each Language Its Tokenizer: Modular Tokenizers for Efficient Multilingual LLMs*, arXiv:2609.15528 — https://arxiv.org/html/2609.15528v1

### Architectures alternatives
25. Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752 — https://arxiv.org/html/2312.00752
26. Beck et al., *xLSTM 7B: A Recurrent LLM for Fast and Efficient Inference*, arXiv:2503.13427 — https://arxiv.org/pdf/2503.13427

---

## Annexe A — Formules utiles

```
Paramètres (Llama-like, tied) :
  P = V·d + L·(2d² + 2d²·n_kv/n_h + 3·d·f + 2d)

Poids (octets) :
  W = P · bpw / 8

KV (octets) :
  KV = 2 · L · n_kv · (d/n_h) · ctx · octets_par_élément

RSS :
  RSS = W + KV + runtime + scratch

Énergie incrémentale par token :
  E ≈ W · 80 pJ/octet + 2·P_non_embed · e_MAC

Énergie murale :
  E_murale ≈ P_plateforme / débit

Entraînement :
  FLOPs = 6 · N · D
  Mémoire Adam fp32 = 16 octets/paramètre
  Mémoire Adam 8-bit = 10 octets/paramètre
```

## Annexe B — Glossaire

- **QAT** : quantization-aware training. On simule la quantification basse précision pendant
  l'entraînement, ce qui permet au modèle d'apprendre à y résister.
- **PTQ** : post-training quantization. On quantifie après coup. Moins cher, mais s'effondre sous 4 bits.
- **GQA / MQA** : grouped-query / multi-query attention. Plusieurs têtes de requête partagent une
  même tête K/V. Réduit le KV cache sans presque rien perdre.
- **KV cache** : mémoire des clés/valeurs d'attention pour les tokens déjà traités. Croît avec le contexte.
- **Ternaire (1,58 bit)** : poids ∈ {−1, 0, +1}. `log2(3) ≈ 1,58` bit d'information par poids.
- **Deep-thin** : architecture profonde et étroite. Meilleure que large et courte à taille égale.
- **Distillation** : un petit modèle apprend les distributions d'un grand.
- **Tied embeddings** : la matrice d'entrée et celle de sortie partagent les mêmes poids.
- **Race-to-idle** : finir plus vite consomme moins d'énergie totale qu'exécuter lentement à
  puissance presque égale.
- **RSS / PSS** : mesure de mémoire résidente d'un processus.
- **Chinchilla** : recette ~20 tokens par paramètre pour un compute-optimal.
