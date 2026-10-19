# Project LLM v0.2 — 200 Mo, généraliste, déploiement web

**Date** : 2026-10-19
**Statut** : révision du rapport v0.1 après précision des contraintes
**Remplace** : les recommandations de `RECHERCHE-LLM-100MB.md` §0, §6 et §9
**Corrige** : la première version de ce document, qui utilisait une arithmétique bits/poids
optimiste. Les tailles ci-dessous sont des **tailles GGUF mesurées sur Hugging Face**, pas des
estimations.

---

## 0. Ce qui change

| Paramètre | v0.1 (hypothèse) | v0.2 (confirmé) |
|---|---|---|
| RAM d'inférence | ~100 Mo | **~100 Mo, extensible à 200 Mo** |
| Entraînement | ≤ 1 Go | **≤ 1 Go de RAM** |
| Périmètre | spécialisé envisagé | **généraliste** |
| Matériel | non précisé | tout ; PC aujourd'hui, **petits PC en serveurs web** |
| Objectif final | modèle | **site web déployé** |

### Le fait qui commande tout

À cette échelle, **le vocabulaire domine le fichier**, pas les paramètres du transformer.

```
Gemma 3 270M : 262 144 tokens × 640 dimensions = 167,8 M paramètres d'embeddings
               soit 62 % des 270 M de paramètres annoncés
               et ~138 Mo du fichier, même en Q6_K
```

C'est pourquoi un modèle « 270 M » pèse 253 Mo en Q4_K_M, et pourquoi Q2_K ne descend qu'à 237 Mo :
la quantification agressive porte sur le transformer, pas sur les embeddings. **Le levier n° 1
n'est donc pas la quantification, c'est la taille du vocabulaire.**

---

## 1. Qui tient réellement dans 200 Mo ?

Tailles de fichiers mesurées (unsloth, DevQuasar), KV en q8, runtime 26 Mo (20 Mo moteur + 6 Mo
scratch).

| Modèle | Fichier | KV 2k | KV 4k | KV 8k | **RSS 2k** | **RSS 4k** | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **MobileLLM-R1-140M Q4_K_M** | 132 Mo | 11,8 | 23,6 | 47,2 | **170 Mo** | **182 Mo** | ✅ le meilleur compromis |
| **SmolLM2-135M Q4_K_M** | 105 Mo | 23,6 | 47,2 | 94,4 | **155 Mo** | **178 Mo** | ✅ mais anglais et faible |
| SmolLM2-135M Q5_K_M | 112 Mo | 23,6 | 47,2 | 94,4 | 162 Mo | 185 Mo | ✅ |
| SmolLM2-135M Q8_0 | 145 Mo | 23,6 | 47,2 | 94,4 | 195 Mo | 218 Mo | ⚠️ 2 k seulement |
| **Gemma 3 270M Q4_K_M** | **253 Mo** | 7,1 | 11,8 | 16,5 | **286 Mo** | **291 Mo** | ❌ **hors budget** |
| Gemma 3 270M IQ4_XS | 241 Mo | 7,1 | 11,8 | 16,5 | 274 Mo | 279 Mo | ❌ |
| Gemma 3 270M Q2_K | 237 Mo | 7,1 | 11,8 | 16,5 | 270 Mo | 275 Mo | ❌ |
| SmolLM2-360M Q2_K | 219 Mo | 41,9 | 83,9 | — | 287 Mo | 329 Mo | ❌ |
| SmolLM2-360M Q4_K_M | 271 Mo | 41,9 | 83,9 | — | 339 Mo | 381 Mo | ❌ |
| **Gemma 3 270M, vocab ramené à 32 k** | ~106 Mo* | 4,1 | 8,2 | 16,5 | **136 Mo** | **141 Mo** | ✅ mais chirurgie requise |

\* calculé, pas mesuré : 253 Mo − (262 144 − 32 768) × 640 octets (embeddings Q8) ≈ 106 Mo.

### Conclusion de la section

**Le relèvement de 100 à 200 Mo ne débloque pas le généraliste 270 M prêt à l'emploi.** Il débloque
un 135–140 M en Q4 avec un contexte utile, ou un 270 M **après chirurgie du vocabulaire**.

Trois options, par ordre de risque croissant :

1. **MobileLLM-R1-140M Q4_K_M** — 170 Mo à 2 k, 182 Mo à 4 k.
   4,2 billions de tokens d'entraînement, 128 k de vocabulaire, très bon en maths/code pour sa
   taille, contexte natif large. C'est le meilleur rapport qualité/risque **immédiat**.
   Réserve : il est affiné pour le raisonnement (maths/code), pas pour la conversation générale —
   mais il est plus facile à réorienter qu'un modèle sous-entraîné.
2. **SmolLM2-135M Q4_K_M** — 155 Mo à 2 k, 178 Mo à 4 k. Le plus léger, le plus documenté,
   mais principalement anglais et faible en raisonnement (GSM8K 1,4 %). Utile comme plancher de
   comparaison, pas comme produit francophone.
3. **Gemma 3 270M + vocabulaire réduit à 32 k** — 136–141 Mo de RSS, le meilleur modèle de base
   (6 000 Md de tokens, 140+ langues), mais il faut **élaguer les embeddings, réentraîner la couche
   de sortie et faire une phase de guérison**. C'est un vrai travail de recherche, pas une commande.

### Pourquoi Gemma 3 270M reste intéressant malgré tout

Son architecture est excellente pour l'inférence contrainte :

```
18 couches · hidden 640 · 4 têtes Q · 1 seule tête KV · head_dim 256
FFN 2048 · 15 couches à fenêtre glissante (512) + 3 globales
→ 2 Ko de KV par token seulement (contre 11,5 Ko pour SmolLM2-135M)
```

À 32 k de contexte, son KV ne coûte que ~64 Mo. Aucun autre modèle de cette taille ne fait mieux.
**Son seul défaut est un vocabulaire de 262 144 tokens, hérité du multilingue.** Si l'on n'a besoin
que du français, de l'anglais et du code, 32 768 tokens suffisent — et le modèle devient
immédiatement déployable sous 200 Mo, avec 5× plus de contexte que SmolLM2 pour un coût mémoire
inférieur.

---

## 2. Entraînement : ce que 1 Go permet vraiment

### 2.1 L'arithmétique

| Méthode sur un modèle de 140–270 M | Poids de base | Optimiseur | Total sans activations | Sous 1 Go ? |
|---|---:|---:|---:|---|
| Full fine-tuning, Adam fp32 | 0,54–1,07 Go | 1,6–3,2 Go | 2,1–4,3 Go | ❌ |
| Full fine-tuning, Adam 8-bit | 0,54–1,07 Go | 0,8–1,6 Go | 1,3–2,7 Go | ❌ |
| LoRA, base bf16 | 0,27–0,54 Go | ~0,03–0,06 Go | 0,3–0,6 Go | ⚠️ avec activations, non |
| **QLoRA, base NF4** | **0,07–0,13 Go** | **~0,02–0,03 Go** | **0,09–0,16 Go** | ✅ |

Avec les activations (batch 1, séquence 512–1 024, gradient checkpointing) et, sur GPU, le contexte
CUDA (200–400 Mo) :

- **sur CPU** : 300–500 Mo de RSS → tient largement sous 1 Go ;
- **sur GPU 1 Go** : 500–900 Mo → tient, sans marge ;
- **sur GPU ≥ 2 Go** : confortable.

### 2.2 La règle

> **Sous 1 Go, on peut *adapter* un modèle de 140–270 M, mais pas *pré-entraîner* un modèle de
> 270 M.** Le pré-entraînement from scratch plafonne autour de **60 M de paramètres**
> (600 Mo d'optimiseur Adam 8-bit + ~250 Mo d'activations).

### 2.3 Temps et coût

| Run | FLOPs | CPU 30 GFLOPS | GPU T4 ~8 TFLOPS | Coût GPU loué |
|---|---:|---:|---:|---:|
| QLoRA SFT, 5 M tokens (140 M) | 8,0 × 10¹⁵ | ~3 jours | ~20 min | < 0,50 $ |
| QLoRA distillation, 20 M tokens | 3,2 × 10¹⁶ | ~12 jours | ~1,1 h | ~1 $ |
| Pré-entraînement 60 M, 1,2 Md tokens | 4,3 × 10¹⁷ | ~166 jours | ~15 h | ~10 $ |
| Pré-entraînement 270 M, 2 Md tokens | 3,2 × 10¹⁸ | ~3,4 ans | ~110 h | ~70 $ |

Un QLoRA sérieux est donc un run de **quelques heures** pour **1 à 2 $**.

---

## 3. La recette

### 3.1 Option A — sans recherche : MobileLLM-R1-140M

```
1. Base     : facebook/MobileLLM-R1-140M (ou base)
2. Données  : 5 à 10 M tokens de français général + domaine
3. QLoRA    : NF4, r=16, alpha=32, seq 1 024, batch 1, LR 1e-4
4. Fusion   : merge en bf16
5. Export   : Q4_K_M (fichier mesuré 132 Mo)
6. Mesure   : RSS, tok/s, qualité — avant mise en ligne
```

### 3.2 Option B — la vraie recherche : Gemma 3 270M à vocabulaire réduit

```
1. Sélection de vocabulaire : garder les 32 768 tokens les plus fréquents
   en français + anglais + code, mesurés sur le corpus cible.
2. Chirurgie : tronquer token_embd et lm_head ; réindexer ; conserver
   les tokens spéciaux (BOS, EOS, <start_of_turn>, <image_soft_token>...).
3. Guérison : 200 à 500 M tokens (français + anglais), LoRA sur toutes
   les couches + embeddings dégelés, LR faible.
4. Export Q4_K_M → fichier attendu ~106 Mo (à mesurer).
5. Évaluation : comparer au modèle d'origine sur 3 jeux ; accepter une
   perte ≤ 3 points.
```

**Pourquoi c'est le bon pari de recherche :** le gain est structurel (62 % du fichier), il ne
dépend d'aucun kernel exotique, et il compose avec tout le reste. Il rejoint l'ADR-002 de la v0.1
(vocab 32 768, embeddings partagés), qui était juste — mais pour la mauvaise raison : ce n'était pas
une optimisation de calcul, c'était **la** condition de tenue en mémoire.

### 3.3 Données

- 5 à 20 M tokens suffisent pour le style, le format, le français et le ton.
- Ils **ne suffisent pas** à ajouter des connaissances : les faits vont dans le RAG.
- Composition : 40 % instructions générales françaises, 30 % domaine, 20 % format structuré,
  10 % refus et « je ne sais pas ».
- Générer avec un grand modèle (distillation) est plus rentable que collecter du web.
- Garder 5 % de données réelles pour éviter l'uniformisation du style.

### 3.4 Évaluation avant déploiement

Trois jeux de 200 à 500 exemples : tâche (exactitude binaire), français (grammaire, registre),
sécurité (hors-sujet, injection, refus).
Critère : **≥ 70 % sur la tâche, ≥ 90 % de format valide, 0 fuite de prompt système.**

---

## 4. Déploiement web sur petits PCs

### 4.1 Contrainte de dimensionnement

Le KV cache est un **pool partagé** : `--ctx-size` est le total pour tous les slots, pas la taille
par slot. À 200 Mo, cela donne :

| Modèle | Fichier | KV/token | Contexte total max (~175 Mo de marge) | Slots réalistes |
|---|---:|---:|---:|---|
| SmolLM2-135M Q4_K_M | 105 Mo | 11,5 Ko | ~4 000 tokens | 2 × 2 k |
| MobileLLM-R1-140M Q4_K_M | 132 Mo | 5,8 Ko | ~7 000 tokens | 4 × 1,75 k |
| Gemma 270M vocab 32 k | ~106 Mo | 2,0 Ko | ~32 000 tokens | 4 × 8 k |

**Gemma à vocabulaire réduit est le seul à offrir un contexte confortable sous 200 Mo.** C'est un
argument produit décisif : un assistant de site web a besoin de 4 à 8 k de contexte par utilisateur.

### 4.2 Architecture logicielle

```
Navigateur → HTTPS + SSE → Caddy/nginx (TLS, rate limit)
          → cache applicatif SQLite (questions fréquentes)
          → llama-server (API compatible OpenAI)
          → GGUF sur disque en mmap
```

Un seul processus par nœud, poids partagés entre les slots, `--cont-batching` actif.

### 4.3 Configuration de référence (MobileLLM-R1-140M)

```bash
llama-server \
  --model /models/mobilellm-r1-140m-q4_k_m.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 7168 --parallel 4 \
  --cont-batching --threads 4 \
  --batch-size 1024 --ubatch-size 256 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --metrics --api-key file:/etc/llama/api-key
```

| Poste | Valeur |
|---|---|
| Poids | 132 Mo |
| KV (7 168 total, q8) | 41,3 Mo |
| Runtime + scratch | 26 Mo |
| **RSS** | **~199 Mo** |
| Slots | 4 × 1 792 tokens |
| Débit agrégé (N100) | ~30–40 tok/s |
| Débit par utilisateur | ~8–10 tok/s |

### 4.4 Matériel

| Plateforme | RAM | Débit mono-flux [estim.] | Utilisateurs | Puissance |
|---|---:|---:|---:|---:|
| Raspberry Pi 5 | 8 Go | 8–12 tok/s | 1–2 | 5–8 W |
| Mini-PC N100 | 16 Go | 18–25 tok/s | 2–4 | 10–15 W |
| Mini-PC Ryzen 5 5500U / i5-1235U | 16 Go | 35–50 tok/s | 6–10 | 15–25 W |

**Pas de GPU.** À 135–270 M de paramètres, le décodage est limité par la bande passante mémoire.
Un GPU n'apporte presque rien et coûte 10× plus cher.

### 4.5 Dimensionnement et énergie

Capacité d'un nœud à 4 slots et ~25 tok/s agrégés : environ **1 requête toutes les 8 s**, soit
~10 000 requêtes/jour en pointe continue. Au-delà, réplicas sans état derrière nginx.

Pour 1 000 requêtes/jour de 200 tokens sur un N100 (15 W) : ~5,8 h de calcul, ~87 Wh/jour,
**~8 €/an** à 0,25 €/kWh. Le serveur coûte plus cher en veille qu'en calcul : réveil à la demande.

### 4.6 Sécurité

Clé API sur llama-server (jamais exposé), rate limit par IP, entrée ≤ 2 000 caractères, sortie
≤ 200–300 tokens, timeout 60 s, prompt système court (200–400 tokens), historique tronqué,
filtre de sortie, `MemoryMax=256M` sous systemd, alerte si RSS > 200 Mo.

---

## 5. Qualité attendue

| Sait faire | Ne sait pas faire |
|---|---|
| Questions simples et courtes | Raisonnement en plusieurs étapes |
| Reformuler, résumer un court texte | Texte long cohérent |
| Traduire des phrases simples | Garantir des faits sans RAG |
| Suivre un format (JSON, liste) | Calcul fiable |
| Conversation de 2–3 tours | Mémoire d'un long historique |

**Un site qui présente ce modèle comme un ChatGPT décevra.** Un site qui l'utilise comme interface
en langage naturel au-dessus d'une base de connaissances (RAG + réponses courtes + sources) sera
utile. Garde-fous : RAG obligatoire pour le factuel, réponses ≤ 200 tokens avec sources, boutons
plutôt que champ libre, avertissement, escalade humaine si le retrieval est faible.

---

## 6. Plan d'expériences

| # | Expérience | Critère go/no-go | Effort |
|---|---|---|---|
| **E0** | RSS réel de MobileLLM-R1-140M Q4_K_M et SmolLM2-135M Q4_K_M dans llama.cpp | écart < 10 % vs calcul | 1 j |
| **E1** | Débit mono-flux et 4 slots sur le mini-PC cible | ≥ 8 tok/s/utilisateur | 1 j |
| **E2** | QLoRA français sur MobileLLM, 5–10 M tokens | +10 pts en win rate humain | 3 j |
| **E3** | RAG + garde-fous | ≥ 70 % sur la tâche | 3 j |
| **E4** | Comparaison à l'aveugle MobileLLM / SmolLM2 / Gemma tronqué | gagnant net sur 3 jeux/4 | 3 j |
| **E5** | **Chirurgie du vocabulaire Gemma 3 : 262 k → 32 k** | perte ≤ 3 pts, fichier ≤ 110 Mo | 2 sem. |
| **E6** | Distillation 140–270 M → 60 M | 60 M ≥ 90 % du parent sur la tâche | 2 sem. |

**Ordre : E0 → E1 → E2/E3 → E5 → E4 → E6.**
E5 est le cœur scientifique du projet : c'est lui qui rend un vrai généraliste déployable sous
200 Mo. E6 est l'option « sous 100 Mo ».

---

## 6 bis. Intégration des autres agents (boucle du 2026-10-19)

Deux agents ont produit en parallèle des résultats qui changent la lecture du budget.

**MNÉMOSYNE — le working set, pas le nombre de paramètres.**
`RSS ≈ N_résidents · b/8 + KV + runtime + cache`. Le reste est du stockage disque.
Mesuré chez eux : charger une cartouche de 2–4 Mo coûte **1–4 ms** (p99 ~100 ms sur VM partagée) ;
le chargement par requête est négligeable ; la pagination par token d'un cœur de 34 Mo coûterait
68 ms/token. D'où leur règle : *cartouche par requête, cœur résident*.
Leur configuration calculée : 60 M + bibliothèque de 1 Md de paramètres ternaires = **69,2 Mo de
RSS** pour 284 Mo de disque, la bibliothèque pouvant croître sans changer le RSS.

**KAIROS — Programme II.**
À 400 B de paramètres, 1 Go de compétences vérifiées rapporte autant qu'1 Go de poids
(0,0004 point/Go). Transposition partielle à notre échelle, mais le principe tient : ce qui est
vérifiable doit être stocké, pas appris.

**La convergence.** Trois chemins indépendants — réduire le coût fixe (vocabulaire), déplacer la
capacité vers des cartouches, déplacer la capacité vers des compétences vérifiées — aboutissent à
la même conclusion : **pour un petit budget, la capacité doit sortir des poids**.

**La limite.** Cette stratégie excelle sur des tâches récurrentes et vérifiables, pas sur la
conversation ouverte. Or le brief demande un généraliste. D'où deux modes assumés (ADR-P-012) :
conversation par le dense + RAG, efficacité par les cartouches sur la tâche récurrente.

**Point à trancher — définition du budget.** RSS hors page cache, ou RAM totale page cache inclus ?
Position : mesurer **les deux** (`VmRSS` et `memory.current` d'un cgroup à 200 Mo). Un GGUF mmapé de
132 Mo tient en VmRSS mais force des évictions sous un plafond cgroup. Un chiffre unique est trompeur.

---

## 7. Décisions

**ADR-006 — Cible d'inférence portée à 200 Mo de RSS.**
Conséquence : un 135–140 M en Q4 devient déployable ; Gemma 3 270M reste hors budget **tel quel**
(253 Mo de fichier) et n'entre dans la cible qu'après réduction du vocabulaire.

**ADR-007 — Pas de pré-entraînement from scratch pour le produit.**
Conséquence : QLoRA d'un modèle pré-entraîné ; le from-scratch devient un sujet de recherche (E6).

**ADR-009 — La taille du vocabulaire est le levier mémoire n° 1.**
Constat mesuré : 62 % des paramètres de Gemma 3 270M sont des embeddings (262 144 × 640).
Conséquence : toute architecture cible doit utiliser un vocabulaire de 32 k, et E5 (élagage +
guérison) est prioritaire sur toute autre optimisation de compression.

**ADR-008 — Déploiement via llama-server, pool de contexte partagé.**
Conséquence : le nombre de slots se déduit du budget mémoire restant, pas l'inverse.

**ADR-P-010 — Le budget 200 Mo est un working set résident, pas une taille de modèle.**
Conséquence : toute proposition est jugée sur ses octets résidents, pas sur ses octets stockés.
Un 60 M + bibliothèque peut concourir ; un 270 M ne concourt pas s'il réside entier.

**ADR-P-011 — La capacité externalisée est une piste produit conditionnelle.**
Condition : un test modulaire vs dense à RSS égal (type E-M2) doit montrer +5 pts à ≤ 2× la
latence. Tant qu'il n'a pas réussi, la baseline reste le dense MobileLLM-R1-140M.

**ADR-P-012 — Deux modes assumés : conversation et compétences.**
Le produit ne promet pas le généraliste par les cartouches. Il promet le généraliste par le dense +
RAG, et l'efficacité par les cartouches sur les tâches récurrentes.

---

## 8. Risques

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| MobileLLM déçoit en conversation française | moyenne | élevé | E4 à l'aveugle ; prévoir QLoRA français |
| La chirurgie de vocabulaire dégrade trop | moyenne | moyen | accepter ≤ 3 pts ; sinon rester sur MobileLLM |
| Débit insuffisant sur N100 | moyenne | élevé | E1 avant achat ; viser Ryzen 5 |
| Attente « ChatGPT » | élevée | élevé | cadrer le produit : RAG + réponses courtes |
| Fuite de prompt système | moyenne | moyen | filtre de sortie, tests adverses |
| Licence Gemma | certaine | moyen | vérifier les termes ; SmolLM2 en secours |

---

## 9. Ce qu'il faut retenir

1. **200 Mo ne donnent pas un généraliste 270 M prêt à l'emploi.** Gemma 3 270M pèse 253 Mo de
   fichier et ~290 Mo de RSS. Le budget est dépassé de 45 %.
2. **La raison est le vocabulaire, pas la quantification** : 262 144 tokens × 640 dimensions =
   167,8 M paramètres, 62 % du modèle, ~138 Mo du fichier.
3. **Sous 200 Mo aujourd'hui** : MobileLLM-R1-140M Q4_K_M (170–182 Mo) ou SmolLM2-135M Q4_K_M
   (155–178 Mo).
4. **La voie vers un vrai généraliste sous 200 Mo** : ramener le vocabulaire de Gemma 3 à 32 k.
   Le fichier tombe à ~106 Mo, le KV à 2 Ko/token, et 32 k de contexte deviennent possibles.
5. **1 Go suffit pour adapter, pas pour pré-entraîner** : QLoRA, 5–20 M tokens, quelques heures,
   1–2 $.
6. **Le déploiement web est bon marché** : un N100, 4 slots, ~8–10 tok/s par utilisateur, ~8 €/an.

---

## Annexe — commandes

```bash
python tools/llm_budget.py table
python tools/llm_budget.py show mobilellm-r1-140m
python tools/llm_budget.py serve mobilellm-r1-140m --parallel 4 --ctx-per-slot 1792
python tools/llm_budget.py wall mobilellm-r1-140m --device-watts 15

# E0 : mesure réelle
/usr/bin/time -v llama-server -m mobilellm-r1-140m-q4_k_m.gguf -c 7168 -np 4 -t 4 &
watch -n 1 'grep VmRSS /proc/$(pgrep llama-server)/status'
```
