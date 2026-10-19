# Project LLM v0.2 — 200 Mo, généraliste, déploiement web

**Date** : 2026-10-19
**Statut** : mise à jour du rapport v0.1 après précision des contraintes
**Remplace** : les recommandations de `RECHERCHE-LLM-100MB.md` §0, §6 et §9
**Complète** : `RECHERCHE-LLM-100MB.md` (l'arithmétique et l'état de l'art restent valides)

---

## 0. Ce qui change

| Paramètre | v0.1 (hypothèse) | v0.2 (confirmé) | Conséquence |
|---|---|---|---|
| RAM d'inférence | ~100 Mo | **~100 Mo, extensible à 200 Mo** | Un vrai généraliste devient possible |
| Entraînement | ≤ 1 Go | **≤ 1 Go de RAM** | QLoRA oui, pré-entraînement 270 M non |
| Périmètre | spécialisé envisagé | **généraliste** | Il faut une base pré-entraînée, pas un modèle from scratch |
| Matériel | non précisé | **tout, PC aujourd'hui, petits PC en serveurs web** | Le débit et la concurrence deviennent centraux |
| Objectif final | modèle | **site web déployé** | L'architecture système compte autant que le modèle |

**Le renversement principal :** à 100 Mo, il fallait construire un petit modèle spécialisé. À 200 Mo,
un modèle généraliste existe déjà et tient — **Gemma 3 270M en QAT INT4**. Le projet doit donc
**fine-tuner et déployer**, pas pré-entraîner. Le pré-entraînement from scratch sous 1 Go plafonne à
~60 M de paramètres, soit un modèle bien plus faible que Gemma 3 270M pour un coût de recherche
incomparablement plus élevé.

> **Recommandation : arrêter la piste « entraîner un LLM depuis zéro » comme piste produit.
> La conserver comme piste de recherche (distillation 270 M → 60 M), pas comme chemin critique.**

---

## 1. Qui tient réellement dans 200 Mo ?

Tous les chiffres sont calculés par `tools/llm_budget.py`, avec le KV cache réel (fenêtre glissante
prise en compte pour Gemma), un runtime de 20 Mo et 6 Mo de scratch.

| Modèle | Params | Poids | KV | **RSS** | Verdict |
|---|---:|---:|---:|---:|---|
| **Gemma 3 270M IT, QAT q4, 2 k ctx** | 268 M | 134,0 Mo | 7,1 Mo | **167 Mo** | ✅ marge 33 Mo |
| **Gemma 3 270M IT, QAT q4, 8 k ctx** | 268 M | 134,0 Mo | 16,5 Mo | **177 Mo** | ✅ **le bon réglage** |
| **Gemma 3 270M IT, QAT q4, 16 k ctx** | 268 M | 134,0 Mo | 29,1 Mo | **189 Mo** | ✅ marge 11 Mo |
| Gemma 3 270M IT, QAT q4, 32 k ctx | 268 M | 134,0 Mo | 54,3 Mo | 214 Mo | ❌ |
| SmolLM2-135M, Q8 | 135 M | 142,9 Mo | 23,6 Mo | 193 Mo | ✅ mais anglais et plus faible |
| SmolLM2-135M, Q4.5 | 135 M | 75,7 Mo | 23,6 Mo | 125 Mo | ✅ marge, mais qualité faible |
| SmolLM2-360M, Q4.5 | 362 M | 203,5 Mo | 41,9 Mo | 271 Mo | ❌ |
| MobileLLM-R1-140M, Q4.5 | 140 M | 78,8 Mo | 11,8 Mo | 117 Mo | ✅ mais spécialisé maths/code |
| Ternaire-400M (pari), 2 k ctx | 394 M | 77,9 Mo | 33,6 Mo | 137 Mo | ✅ mais non entraînable sous 1 Go |

### Pourquoi Gemma 3 270M gagne

Sa configuration exacte (config officielle) :

```
18 couches · hidden 640 · 4 têtes Q · 1 tête KV · head_dim 256
FFN 2048 · vocab 262 144 · embeddings partagés
15 couches à fenêtre glissante (512) + 3 couches globales
contexte natif 32 768 · 6 000 Md de tokens · 140+ langues
```

Trois propriétés décisives :

1. **Une seule tête KV.** Le KV par token est de 512 octets par couche en q8, soit ~2,5 Ko/token
   en moyenne avec la fenêtre glissante. À 8 k de contexte, cela ne coûte que **16,5 Mo**.
   C'est ce qui rend le généraliste compatible avec 200 Mo.
2. **La fenêtre glissante 5:1.** 15 couches sur 18 ne stockent que 512 tokens. Sans elle, le même
   modèle à 8 k coûterait ~40 Mo de KV.
3. **La QAT officielle.** Google publie un checkpoint entraîné à supporter l'INT4
   (`gemma-3-270m-it-qat-q4_0-unquantized`). À 4 bits, un 270 M coûte 134 Mo au lieu de 536 Mo en bf16.

**Point de vigilance :** les GGUF `Q4_K_M` de ce modèle pèsent ~250 Mo, car ils gardent les
embeddings en Q8 (170 M de paramètres = 170 Mo à eux seuls). Pour tenir sous 200 Mo, il faut un
**Q4_0 / QAT sur l'ensemble des tenseurs, embeddings compris**. C'est le seul réglage qui donne 134 Mo.

### Pourquoi pas SmolLM2

SmolLM2 est entraîné principalement en anglais. Pour un site francophone, Gemma 3 (140+ langues) est
un meilleur point de départ, même à taille comparable. SmolLM2-360M, qui serait qualitativement
intéressant, ne rentre pas (271 Mo).

---

## 2. Entraînement : ce que 1 Go permet vraiment

### 2.1 L'arithmétique, sans illusion

| Méthode sur Gemma 3 270M | Poids de base | Optimiseur | Total sans activations | Tient sous 1 Go ? |
|---|---:|---:|---:|---|
| Full fine-tuning, Adam fp32 | 1,07 Go | 3,2 Go | 4,3 Go | ❌ |
| Full fine-tuning, Adam 8-bit | 1,07 Go | 1,6 Go | 2,7 Go | ❌ |
| LoRA, base bf16 | 0,54 Go | ~0,06 Go | 0,6 Go | ⚠️ avec activations, non |
| **QLoRA, base NF4** | **0,134 Go** | **~0,03 Go** | **0,16 Go** | ✅ |

Avec les activations (batch 1, séquence 512 à 1 024, gradient checkpointing) et, sur GPU, le contexte
CUDA (200 à 400 Mo) :

- **sur CPU** : ~350 à 500 Mo de RSS → **tient largement sous 1 Go** ;
- **sur GPU 1 Go** : ~600 à 900 Mo → tient, mais sans marge ;
- **sur GPU ≥ 2 Go** : confortable.

### 2.2 La règle à retenir

> **Sous 1 Go, on peut *adapter* un modèle de 270 M, mais pas *pré-entraîner* un modèle de 270 M.**
> Le pré-entraînement from scratch plafonne autour de **60 M de paramètres** (600 Mo d'optimiseur
> Adam 8-bit + ~250 Mo d'activations), et un 60 M généraliste est très inférieur à Gemma 3 270M.

### 2.3 Ce que ça coûte en temps

| Run | FLOPs | CPU 30 GFLOPS | GPU T4 ~8 TFLOPS | Coût GPU loué |
|---|---:|---:|---:|---:|
| QLoRA SFT, 5 M tokens | 8,0 × 10¹⁵ | ~3 jours | ~20 min | < 0,50 $ |
| QLoRA distillation, 20 M tokens | 3,2 × 10¹⁶ | ~12 jours | ~1,1 h | ~1 $ |
| Pré-entraînement 60 M, 1,2 Md tokens | 4,3 × 10¹⁷ | ~166 jours | ~15 h | ~10 $ |
| Pré-entraînement 270 M, 2 Md tokens | 3,2 × 10¹⁸ | ~3,4 ans | ~110 h | ~70 $ |

**Conclusion opérationnelle :** un QLoRA sérieux (5 à 20 M tokens) est un run de **quelques heures**
sur un GPU loué, pour **1 à 2 $**. C'est le seul entraînement qui respecte à la fois le budget
mémoire, le budget temps et le budget argent.

---

## 3. La recette : Gemma 3 270M, français, domaine, sous 1 Go

### 3.1 Chaîne complète

```
1. Base     : google/gemma-3-270m-it-qat-q4_0-unquantized
              (checkpoint QAT, poids "unquantized" mais robustes à l'INT4)
2. Données  : 5 à 20 M tokens de français de qualité + domaine
              (instructions, dialogues courts, format JSON, refus)
3. QLoRA    : NF4, r=16, alpha=32, cibles q/k/v/o + gate/up/down
              seq 1 024, batch 1, gradient checkpointing, LR 1e-4
4. Fusion   : merge des adaptateurs dans les poids bf16
5. Export   : Q4_0 (tous tenseurs à 4 bits, embeddings compris) ou QAT q4_0
6. Mesure   : RSS, tok/s, qualité sur le jeu de test, avant toute mise en ligne
```

**Pourquoi partir du checkpoint QAT plutôt que du modèle fp16 ?** Parce que la QAT apprend au modèle
à vivre avec 4 bits. Fine-tuner le modèle fp16 puis le quantifier à 4 bits après coup cumule deux
pertes : celle du fine-tuning et celle de la PTQ. Partir du QAT réduit la seconde à presque rien.

**Attention :** un QLoRA sur un checkpoint QAT peut dégrader sa robustesse à la quantification si le
learning rate est trop élevé. Utiliser un LR faible (5e-5 à 1e-4) et vérifier la perplexité après
export Q4_0. Si la dégradation dépasse 3 %, refaire un court QAT de récupération.

### 3.2 Données : le vrai goulot d'étranglement

- 5 à 20 M tokens suffisent pour le style, le format, le français et le ton.
- Ils **ne suffisent pas** à ajouter des connaissances. Les faits vont dans le RAG.
- Composition recommandée : 40 % instructions générales françaises, 30 % domaine, 20 % format
  structuré (JSON, tableaux, listes), 10 % refus et « je ne sais pas ».
- Générer les données avec un grand modèle (distillation) est plus rentable que collecter du web.
- Garder 5 % de données réelles pour éviter l'uniformisation du style.

### 3.3 Évaluation avant déploiement

Trois jeux, 200 à 500 exemples chacun :
1. **Tâche** : ce que le site doit savoir faire (exactitude binaire).
2. **Français** : grammaire, accords, registre (évaluation humaine ou juge LLM).
3. **Sécurité** : hors-sujet, prompt injection, demandes sensibles (taux de refus correct).

Critère de mise en ligne : **≥ 70 % sur la tâche, ≥ 90 % de format valide, 0 fuite de prompt système.**

---

## 4. Déploiement web sur petits PCs

### 4.1 Matériel : ce qui est réaliste

| Plateforme | RAM | Débit mono-flux [estim.] | Utilisateurs confortables | Puissance |
|---|---:|---:|---:|---:|
| Raspberry Pi 5 | 8 Go | 5–9 tok/s | 1–2 | 5–8 W |
| Mini-PC N100 | 16 Go | 12–18 tok/s | 3–5 | 10–15 W |
| Mini-PC Ryzen 5 5500U / i5-1235U | 16 Go | 25–40 tok/s | 8–12 | 15–25 W |
| Ancien laptop i5-8250U | 8–16 Go | 15–25 tok/s | 4–8 | 15 W |

**Le Raspberry Pi 5 est possible mais lent** (≈ 50 s pour une réponse de 200 tokens à 4 slots).
Un N100 est le meilleur rapport performance/prix/consommation pour démarrer. Un Ryzen 5 offre
2 à 3× plus de confort pour ~2× le prix.

**Ne pas acheter de GPU.** À 270 M de paramètres, le décodage est limité par la bande passante
mémoire, pas par le calcul. Un GPU n'apporte presque rien et coûte 10× plus cher.

### 4.2 Architecture logicielle

```
Navigateur
  │  HTTPS + SSE (streaming)
  ▼
Caddy / nginx  ── TLS, rate limit, cache HTTP, compression
  │
  ├──► cache applicatif (SQLite) : questions fréquentes, réponses pré-calculées
  │
  ▼
llama-server (llama.cpp)  ── API compatible OpenAI
  │   --ctx-size 8192 --parallel 4 --cont-batching --metrics
  ▼
GGUF sur disque (mmap, lecture seule)
```

**Un seul processus llama-server par nœud**, modèle chargé une fois, poids partagés entre les slots.
Le KV cache est **un pool partagé** : `--ctx-size` est le total pour tous les slots, pas par slot.
Avec `--parallel 4 --ctx-size 8192`, chaque requête dispose de ~2 048 tokens.

### 4.3 Configuration de référence

```bash
llama-server \
  --model /models/gemma-3-270m-it-qat-q4_0.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 8192 \
  --parallel 4 \
  --cont-batching \
  --threads 4 \
  --batch-size 1024 --ubatch-size 256 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --metrics \
  --api-key file:/etc/llama/api-key
```

Mesures attendues [à valider par E0/E1] :

| Poste | Valeur |
|---|---|
| Poids | 134 Mo |
| KV (8 k total, q8) | 16,5 Mo |
| Runtime + scratch | ~26 Mo |
| **RSS** | **~177 Mo** |
| Slots | 4 × ~2 k tokens |
| Débit agrégé (N100) | ~35–40 tok/s |
| Débit par utilisateur | ~9 tok/s |
| Latence, 200 tokens | ~21 s (streaming : premier token en ~1 s) |

**Limite système :** placer le service sous `MemoryMax=256M` (systemd) ou `mem_limit: 256m`
(Docker). Si le processus dépasse, c'est un bug de configuration, pas une fatalité.

### 4.4 Dimensionnement et montée en charge

| Trafic | Nœuds N100 nécessaires | Commentaire |
|---|---:|---|
| 1 000 req/jour | 1 | ~0,01 req/s, largement sous la capacité |
| 10 000 req/jour | 1 | ~0,12 req/s, confortable |
| 100 000 req/jour | 3–4 | ~1,2 req/s, avec réplicas |
| 1 M req/jour | 20–30 | il faut passer à un vrai serveur |

Capacité d'un nœud : 4 slots / 21 s ≈ **0,19 req/s ≈ 16 000 req/jour** en pointe continue.
Au-delà, ajouter des réplicas derrière nginx (round-robin). Les réplicas sont **sans état** :
la session est côté client ou dans le cache, pas dans le processus.

**Le cache est le levier n° 1.** Sur un site public, 30 à 60 % des questions sont répétées.
Un cache SQLite de 10 000 réponses coûte quelques Mo et divise la charge par 2 à 3.

### 4.5 Énergie et coût

Pour 1 000 requêtes/jour de 200 tokens sur un N100 (15 W) :

```
temps de calcul ≈ 1000 × 21 s = 5,8 h
énergie        ≈ 15 W × 5,8 h = 87 Wh/jour
coût annuel    ≈ 32 kWh ≈ 8 € à 0,25 €/kWh
```

**Le serveur coûte plus cher en veille qu'en calcul.** À 1 000 requêtes/jour, l'énergie est
négligeable. Le vrai enjeu énergétique est de **ne pas laisser le service tourner à vide** :
réveil à la demande, mise en veille la nuit, ou mutualisation sur un seul nœud.

### 4.6 Sécurité et robustesse

- **Clé API** sur llama-server, jamais exposé directement à Internet.
- **Rate limit** par IP (nginx `limit_req`), 10 req/min par défaut.
- **Taille d'entrée** limitée (2 000 caractères) et **sortie** limitée (`max_tokens` 200–300).
- **Timeout** de 60 s ; le client voit le streaming, mais la requête est coupée au-delà.
- **Prompt système court** (200–400 tokens) pour préserver le contexte utile.
- **Historique tronqué** aux 2–3 derniers tours.
- **Filtre de sortie** : refus des sujets sensibles, détection de fuite du prompt système.
- **Sauvegarde** du GGUF et des adaptateurs ; mise à jour atomique par symlink + restart.
- **Supervision** : `/metrics` (RSS, slots occupés, tok/s), logs JSON, alerte si RSS > 200 Mo.

---

## 5. Qualité attendue : être honnête

Avec Gemma 3 270M en Q4 :

| Ce qu'il sait faire | Ce qu'il ne sait pas faire |
|---|---|
| Répondre à des questions simples et courtes | Raisonner en plusieurs étapes |
| Reformuler, résumer un court texte | Produire un texte long cohérent |
| Traduire des phrases simples | Garantir des faits sans RAG |
| Suivre un format (JSON, liste, tableau) | Calculer de façon fiable |
| Tenir une conversation de 2–3 tours | Se souvenir d'un long historique |
| Dire « je ne sais pas » (si entraîné) | Être un « ChatGPT » |

**Un site qui présente ce modèle comme un assistant généraliste décevra.** Un site qui l'utilise
comme **interface en langage naturel au-dessus d'une base de connaissances** (RAG + réponses courtes
+ sources) sera utile.

### Garde-fous produit

1. **RAG obligatoire** pour toute question factuelle.
2. **Réponses courtes** (≤ 200 tokens), avec sources cliquables.
3. **Boutons / suggestions** plutôt que champ libre pour les cas fréquents.
4. **Avertissement** : « assistant automatique, peut se tromper ».
5. **Escalade humaine** quand la confiance est basse (score de retrieval faible).

---

## 6. Plan d'expériences v0.2

| # | Expérience | Question | Critère go/no-go | Effort |
|---|---|---|---|---|
| **E0** | RSS réel de Gemma 3 270M QAT q4 dans llama.cpp | Tient-on sous 200 Mo ? | RSS < 190 Mo à 8 k | 1 j |
| **E1** | Débit mono-flux et 4 slots sur le mini-PC cible | Le site est-il utilisable ? | ≥ 8 tok/s/utilisateur | 1 j |
| **E2** | QLoRA français, 5 M tokens | Gain perceptible vs base ? | +10 pts en win rate humain | 3 j |
| **E3** | RAG + garde-fous | Le système répond-il juste ? | ≥ 70 % sur la tâche | 3 j |
| **E4** | Gemma 270M vs SmolLM2-135M Q8 en français | Le choix est-il confirmé ? | Gemma gagne sur 3 jeux/4 | 2 j |
| **E5** | Contexte 16 k (189 Mo) | A-t-on besoin de plus que 8 k ? | gain réel sur les prompts | 1 j |
| **E6** | Distillation 270M → 60M | Peut-on descendre sous 100 Mo ? | 60M ≥ 90 % du 270M sur la tâche | 2 sem. |
| **E7** | Ternaire 400M | Pari de capacité | — (hors budget d'entraînement) | — |

**Ordre : E0 → E1 → E3 → E2 → E4.** E0 et E1 se font en une journée et décident de tout le reste.

E6 est la seule vraie piste de recherche : elle permettrait un modèle à 100 Mo qui conserve 90 % de
la qualité du 270M. C'est le sujet scientifique du projet, mais il ne doit pas bloquer le produit.

---

## 7. Décisions

**ADR-006 — Cible d'inférence portée à 200 Mo de RSS.**
Conséquence : Gemma 3 270M devient candidate ; SmolLM2-360M reste exclue (271 Mo).

**ADR-007 — Pas de pré-entraînement from scratch pour le produit.**
Le budget de 1 Go plafonne à ~60 M de paramètres, très inférieur à Gemma 3 270M.
Conséquence : QLoRA + distillation ; le from-scratch devient un sujet de recherche (E6).

**ADR-008 — Déploiement via llama-server sur petits PCs, 8 k de contexte total, 4 slots.**
Conséquence : RSS ~177 Mo, 4 utilisateurs simultanés, montée en charge horizontale par réplicas.
Le cache applicatif est obligatoire, pas optionnel.

---

## 8. Risques spécifiques à la v0.2

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| Le GGUF Q4_0 dégrade trop la qualité | moyenne | élevé | utiliser le checkpoint QAT ; mesurer la perplexité |
| Le débit sur N100 est insuffisant | moyenne | élevé | tester E1 avant d'acheter ; viser Ryzen 5 |
| Les utilisateurs attendent ChatGPT | élevée | élevé | cadrer le produit : RAG + réponses courtes |
| Fuite de prompt système | moyenne | moyen | filtre de sortie, tests adverses |
| Coût d'exploitation sous-estimé | faible | faible | 8 €/an à 1 000 req/jour |
| Le 1 Go bloque le QLoRA sur GPU | moyenne | moyen | CPU (lent) ou GPU loué 2 Go |
| Dépendance à un modèle sous licence Gemma | certaine | moyen | vérifier les termes ; garder SmolLM2 en secours |

---

## 9. Ce qu'il faut retenir

1. **200 Mo suffisent pour un généraliste** : Gemma 3 270M QAT q4, 177 Mo, 8 k de contexte.
2. **1 Go suffit pour l'adapter**, pas pour le pré-entraîner : QLoRA, 5 à 20 M tokens, quelques heures
   de GPU, 1 à 2 $.
3. **Le déploiement web est simple et bon marché** : un mini-PC N100, llama-server, 4 slots,
   ~9 tok/s par utilisateur, 8 €/an d'électricité.
4. **La qualité sera celle d'un petit assistant**, pas d'un ChatGPT. Le produit doit être conçu
   autour du RAG et de réponses courtes.
5. **La seule recherche qui vaut le coup** est la distillation 270 M → 60 M, pour passer sous 100 Mo
   sans perdre l'essentiel. Tout le reste est de l'ingénierie.

---

## Annexe — Commandes utiles

```bash
# Budget mémoire et énergie
python tools/llm_budget.py show gemma3-270m-qat-8k
python tools/llm_budget.py serve gemma3-270m-qat-8k --parallel 4 --ctx-per-slot 2048
python tools/llm_budget.py wall gemma3-270m-qat-8k --device-watts 15

# Mesure réelle du RSS (à faire en E0)
/usr/bin/time -v llama-server -m model.gguf -c 8192 -np 4 -t 4 &
watch -n 1 'grep VmRSS /proc/$(pgrep llama-server)/status'
```
