# Journal des agents

Format : voir `AGENTS.md` §4. Ajouts en fin de fichier, jamais de réécriture de l'historique.

---

```
[2026-10-19] Agent-Recherche · W1/W5/W6 · FAIT
Fait      : Rapport de recherche v0.1 + calculateur de budget mémoire/énergie/compute.
Résultat  : À 4,5 bits/poids, ~85 M paramètres maximum pour 100 Mo de RSS avec 2 k de contexte.
            NanoFR-75M : 41,2 Mo de poids + 10,5 Mo de KV + 26 Mo runtime = 78 Mo.
            Énergie incrémentale ~3,5 mJ/token ; énergie murale ~1,5 J/token à 10 tok/s sur 15 W.
Fichiers  : docs/RECHERCHE-LLM-100MB.md, tools/llm_budget.py, AGENTS.md, coordination/*
Bloqueurs : ambiguïté sur « 1 Go » (mémoire du run / données / checkpoint).
Besoin    : validation de l'interprétation RSS pour les 100 Mo.
Décision proposée : ADR-001 à ADR-005.
```

```
[2026-10-19] Agent-Recherche · W1 · À FAIRE
Fait      : —
Résultat  : —
Fichiers  : —
Bloqueurs : aucun
Besoin    : mesure E0 (RSS réel d'un modèle 40 Mo dans llama.cpp) avant de figer l'architecture.
Décision proposée : aucune
```

```
[2026-10-19] Agent-Recherche · W1/W5/W6 · FAIT
Fait      : Révision v0.2 après passage de la cible à 200 Mo et cadrage généraliste + web.
Résultat  : Gemma 3 270M QAT q4 = 134,0 Mo de poids + 7,1/16,5/29,1 Mo de KV (2k/8k/16k)
            + 26 Mo runtime = 167/177/189 Mo de RSS. Le KV est minuscule grâce à 1 seule tête KV
            et 15 couches sur 18 à fenêtre glissante de 512 : 2,5 Ko/token en moyenne.
            Sous 1 Go : QLoRA NF4 (0,16 Go) oui ; full FT Adam 8-bit (2,7 Go) non.
            Déploiement : 1 llama-server, --ctx-size 8192 --parallel 4, ~9 tok/s/utilisateur,
            89 Wh/jour pour 1 000 requêtes de 200 tokens sur N100 → ~8 €/an.
Fichiers  : docs/ARCHITECTURE-200MB-WEB.md (nouveau), README.md, tools/llm_budget.py
            (presets gemma3-270m-qat-{2k,8k,32k}, KV à fenêtre glissante, commande `serve`),
            coordination/decisions.md (ADR-006 à 008)
Bloqueurs : aucun
Besoin    : E0 (RSS réel dans llama.cpp) et E1 (débit sur le mini-PC cible) avant achat de matériel.
Décision proposée : ADR-006, ADR-007, ADR-008.
```

```
[2026-10-19] Agent-Recherche · W7 (déploiement) · À FAIRE
Fait      : —
Résultat  : —
Fichiers  : —
Bloqueurs : aucun
Besoin    : définir la stack du site (front, auth, quota), le cache applicatif, la supervision.
Décision proposée : ouverture du chantier W7 « Produit & déploiement ».
```

```
[2026-10-19] Agent-Recherche · W1/W3 · CORRECTION
Fait      : Vérification des tailles GGUF réelles sur Hugging Face ; correction du document v0.2.
Résultat  : Gemma 3 270M Q4_K_M = 253 Mo (IQ4_XS 241, Q2_K 237) → RSS 286–296 Mo : HORS BUDGET.
            SmolLM2-135M Q4_K_M = 105 Mo → RSS 155 Mo (2k) / 178 Mo (4k).
            MobileLLM-R1-140M Q4_K_M = 132 Mo → RSS 170 Mo (2k) / 182 Mo (4k).
            SmolLM2-360M Q4_K_M = 271 Mo → exclue.
            Cause : 262 144 × 640 = 167,8 M paramètres d'embeddings = 62 % des paramètres.
            Gemma 3 270M avec vocabulaire ramené à 32 k ≈ 106 Mo → RSS 136–141 Mo.
Fichiers  : docs/ARCHITECTURE-200MB-WEB.md (réécrit), tools/llm_budget.py (file_mb mesuré),
            coordination/decisions.md (ADR-006 corrigé, ADR-009 ajouté)
Bloqueurs : aucun
Besoin    : E0 (RSS réel) et E5 (chirurgie du vocabulaire) pour valider.
Décision proposée : ADR-009.
```

```
[2026-10-19] Agent-Recherche · W6 · BOUCLE DE SYNCHRONISATION
Fait      : Relecture complète du doc ; intégration de KAIROS (Programme II) et MNÉMOSYNE
            (working set / ALEXANDRIA) ; réponses écrites dans leurs onglets.
Résultat  : Convergence indépendante des trois agents : « pour un petit budget, la capacité doit
            sortir des poids ». MNÉMOSYNE mesure 1–4 ms pour charger une cartouche de 2–4 Mo.
            ALEX 60 M + 1 Md ternaire = 69,2 Mo de RSS pour 284 Mo de disque [calculé].
            Limite : la stratégie excelle sur des tâches récurrentes, pas sur la conversation
            ouverte — or le brief demande un généraliste.
Fichiers  : AGENTS.md (§1 bis boucle), docs/ARCHITECTURE-200MB-WEB.md (§6 bis),
            coordination/decisions.md (ADR-P-010/011/012)
Bloqueurs : E-M2 (modulaire vs dense à RSS égal) non exécutée ; définition du budget
            (VmRSS vs memory.current) à trancher par l'humain.
Besoin    : réponse de l'humain sur (a) RSS hors page cache ou RAM totale, (b) 1–10 Go de disque
            pour la bibliothèque, (c) autorisation d'instrumenter un site pour mesurer α.
Décision proposée : ADR-P-010, ADR-P-011, ADR-P-012.
```

```
[2026-10-19] Agent-Recherche · W1/W3/W5 · FAIT (correction majeure)
Fait      : Vérification de la config officielle de Gemma 3 270M et des tailles GGUF réelles.
Résultat  : CONFIG RÉELLE : 18 couches, d=640, 4 têtes Q, 1 tête KV, head_dim 256, FFN 2048,
            vocab 262 144, 15 couches locales (fenêtre 512) + 3 globales, ctx 32 k.
            KV = 512 o/token/couche → 7,1 Mo à 2 k, 16,5 Mo à 8 k, 29,1 Mo à 16 k.
            Le Q4_K_M standard pèse 253 Mo parce que les embeddings restent à ~8 bits.
            Avec --token-embedding-type q4_0 (permis par la QAT) : ~150 Mo → RSS 183/193/205 Mo
            à 2 k/8 k/16 k. LE GÉNÉRALISTE TIENT DONC DANS 200 Mo AVEC 8 k DE CONTEXTE.
            Correction des affirmations v0.1 « plancher 300–500 Mo » et v0.2 « hors budget ».
Fichiers  : docs/ARCHITECTURE-200MB-WEB.md (v0.3), tools/llm_budget.py
            (presets gemma3-270m-q4emb*, KV à fenêtre glissante, head_dim explicite),
            docs/PROTOCOLE-EVALUATION.md (nouveau), README.md, coordination/decisions.md
Bloqueurs : E0 (taille et RSS réels du GGUF emb4) et E1 (qualité) non exécutées.
Besoin    : réponse de l'humain au protocole d'évaluation (500 requêtes, domaine cible).
Décision proposée : ADR-P-010.
```

```
[2026-10-19] Agent-Recherche · W5/W6 · FAIT (intégration E-M0b)
Fait      : Intégration de la mesure page cache de MNÉMOSYNE (E-M0b) et réponse à Silex.
Résultat  : Fichier 1,91 GiB lu par blocs : memory.current 55,8 → 2 019,5 Mio, VmRSS 34,7 Mio,
            836 Mio/s. Avec fadvise(DONTNEED) : memory.current 61,8 Mio, 625 Mio/s (−25 %).
            Conséquence : le plafond de 200 Mo doit porter sur le RSS du processus, pas sur le
            cgroup entier, sinon éviction du GGUF et chute du débit.
            Protocole E0 mis à jour : quatre chiffres à publier (RSS, cgroup avec/sans cache,
            débit dans les deux cas).
Fichiers  : docs/PROTOCOLE-EVALUATION.md §7.5, docs/ARCHITECTURE-200MB-WEB.md §4.7
Bloqueurs : E0 non exécutée (pas de llama.cpp ni de GGUF dans cet environnement).
Besoin    : domaine de la tâche cible + autorisation logs pour le jeu de 500 requêtes.
Décision proposée : aucune nouvelle.
```
