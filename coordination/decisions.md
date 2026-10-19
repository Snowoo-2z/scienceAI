# Journal des décisions (ADR)

---

## ADR-001 — Interpréter la cible mémoire comme le RSS du processus d'inférence

**Date** : 2026-10-19
**Statut** : accepté (par défaut, à confirmer par l'utilisateur)
**Contexte** : « ~100 Mo de RAM » peut désigner le fichier de poids, la mémoire du modèle, ou le RSS
du processus. Les trois diffèrent d'un facteur 1,5 à 2.
**Décision** : la cible est le **RSS au pic** du processus d'inférence, incluant poids, KV cache,
runtime et buffers de calcul, hors système d'exploitation.
**Conséquences** : un modèle de 135 M en Q4 (110 Mo de poids) est **hors budget**. La frontière se
situe autour de 85 M paramètres en 4-bit.

---

## ADR-002 — Vocabulaire de 32 768 tokens, embeddings partagés

**Date** : 2026-10-19
**Statut** : proposé
**Contexte** : la table d'embeddings consomme `vocab × d_model` paramètres. À 100 Mo, un vocabulaire
de 128 k ou 256 k consomme 40 à 85 Mo à lui seul.
**Décision** : vocabulaire 32 768, embeddings entrée/sortie partagés (tied).
**Conséquences** : économie d'environ 30 Mo par rapport à un vocabulaire de 128 k. Perte attendue
sur le code et les langues non latines. À valider par l'expérience E2.

---

## ADR-003 — Cible d'architecture : 60–85 M paramètres, 4-bit QAT

**Date** : 2026-10-19
**Statut** : proposé
**Contexte** : à 4,5 bits/poids, la frontière des 100 Mo avec 2 k de contexte est ~85 M paramètres.
**Décision** : architecture de référence NanoFR-75M (20 couches, d=512, 8/2 têtes, FFN 1 408,
vocab 32 k, contexte 2 048, KV q8).
**Conséquences** : RSS estimé 78 Mo, marge de 22 Mo. Le ternaire 200 M reste une piste exploratoire.

---

## ADR-004 — L'énergie est mesurée par réponse utile, pas par token

**Date** : 2026-10-19
**Statut** : accepté
**Contexte** : à petite échelle, `E ≈ P_plateforme / débit`. Un modèle plus petit mais plus lent
peut consommer davantage.
**Décision** : toute comparaison énergétique rapporte `J/réponse juste`, le débit et la puissance de
la plateforme.
**Conséquences** : les optimisations de latence sont aussi des optimisations énergétiques.

---

## ADR-005 — Ne pas utiliser de MoE ni de sparsité pour la cible 100 Mo

**Date** : 2026-10-19
**Statut** : accepté
**Contexte** : en décodage batch=1, le débit CPU est limité par la bande passante mémoire. Un MoE
réduit les FLOPs mais augmente l'empreinte.
**Décision** : architecture dense uniquement, sauf démonstration contraire mesurée.

---

## ADR-006 — Cible d'inférence portée à 200 Mo de RSS

**Date** : 2026-10-19
**Statut** : accepté (précision du demandeur)
**Contexte** : la contrainte initiale de ~100 Mo est explicitement relaxée à ~200 Mo. À cette taille,
un modèle généraliste pré-entraîné devient déployable, ce qui change la nature du projet.
**Décision** : le budget de référence passe à **200 Mo de RSS** (poids + KV + runtime). 100 Mo reste
l'aspiration et l'objectif de la piste de recherche E6.
**Conséquences** : un 135–140 M en Q4 devient déployable (MobileLLM-R1-140M : 170–182 Mo ;
SmolLM2-135M : 155–178 Mo). **Gemma 3 270M reste hors budget tel quel** : 253 Mo de fichier Q4_K_M,
286–296 Mo de RSS. SmolLM2-360M reste exclue (271 Mo de fichier, 339 Mo de RSS).

---

## ADR-007 — Pas de pré-entraînement from scratch pour le produit

**Date** : 2026-10-19
**Statut** : accepté
**Contexte** : sous 1 Go, Adam 8-bit sur un 270 M coûte déjà ~2,7 Go avec les activations. Le
pré-entraînement sous 1 Go plafonne vers 60 M de paramètres (600 Mo d'optimiseur + ~250 Mo
d'activations), soit un modèle très inférieur à Gemma 3 270M pour un coût de recherche bien plus
élevé (E6 : ~166 jours CPU pour 1,2 Md tokens).
**Décision** : le produit repose sur **QLoRA d'un modèle pré-entraîné**. Le from-scratch devient un
sujet de recherche de distillation (E6), pas le chemin critique.
**Conséquences** : le livrable est un pipeline données → QLoRA → merge → export Q4_0 → mesure →
déploiement. Budget : 5–20 M tokens, quelques heures de GPU, 1–2 $.

---

## ADR-008 — Déploiement : un `llama-server` par nœud, 8 k de contexte total, 4 slots

**Date** : 2026-10-19
**Statut** : accepté
**Contexte** : l'objectif final est un site web servi par des petits PC. `llama-server` expose une
API compatible OpenAI, gère le continuous batching et un KV cache unifié. `--ctx-size` désigne le
**pool total** partagé entre les slots, pas la taille par slot.
**Décision** : configuration de référence `--ctx-size 8192 --parallel 4 --cache-type-k/v q8_0`
sur un GGUF Q4_0 de Gemma 3 270M, derrière Caddy/nginx, avec cache applicatif SQLite.
**Conséquences** : RSS ~177 Mo, 4 utilisateurs simultanés, ~9 tok/s chacun sur N100, montée en
charge horizontale par réplicas sans état. Le cache applicatif est obligatoire (30–60 % de questions
répétées sur un site public).

---

## ADR-009 — La taille du vocabulaire est le levier mémoire n° 1

**Date** : 2026-10-19
**Statut** : accepté
**Contexte** : les tailles GGUF mesurées contredisent l'arithmétique bits/poids. Gemma 3 270M :
262 144 tokens × 640 dimensions = 167,8 M paramètres d'embeddings, soit 62 % des paramètres et
~138 Mo du fichier même en Q6_K. C'est pourquoi Q4_K_M pèse 253 Mo et Q2_K seulement 237 Mo.
**Décision** : toute architecture cible utilise un vocabulaire de 32 768 tokens. L'élagage du
vocabulaire de Gemma 3 (262 k → 32 k) + guérison devient l'expérience E5, prioritaire sur toute
autre optimisation de compression.
**Conséquences** : fichier attendu ~106 Mo, KV 2 Ko/token, jusqu'à 32 k de contexte sous 200 Mo.
La piste est structurelle et ne dépend d'aucun kernel exotique.
