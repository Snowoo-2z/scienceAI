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

[2026-10-19] Silex · W3 · REVUE
Fait : présentation dans le Google Doc ; protocole et ADR distants lus ; note de faisabilité rédigée.
Résultat : distinction poids résidents / streaming / système outillé ; borne conditionnelle D/B, aucune performance LLM mesurée.
Fichiers : docs/research/silex-streaming.md, coordination/agents.md.
Bloqueurs : matériel, cible précise et tâche à confirmer.
Besoin : coordination W5/E0 ; revue des hypothèses de streaming ; accès futur aux poids licenciés.
Décision proposée : aucune modification des ADR acceptés ; S1 reste un protocole proposé.

[2026-10-19] Silex · W3 · REVUE — recadrage et S2
Fait : utilisateur confirme 100 Mo, matériel « tout », conversation générale et recherche d'architectures nouvelles plutôt que réduction d'un Transformer. Message transmis immédiatement dans le Doc ; révision des restrictions ADR-003/005 demandée.
Résultat : prototype stdlib W_i=U diag(c(i)) V, 9 tests d'équivalence passent. Pour 8/128/2048 étapes, facteurs 8192 octets fixes, RSS 11,57/11,65/11,65 Mo sur Xeon x86_64 KVM, Python 3.11.2, float64, vecteur synthétique d=64, r=8, seed42, 2026-10-19. Aucun apprentissage ni qualité LLM ni énergie mesurés.
Fichiers : experiments/silex/{shared_operators.py,README.md,result-*.json}.
Bloqueurs : capacité utile non démontrée ; demande d'attribution S3 aux autres agents.
Besoin : KAIROS doit noter que l'unité vient d'être confirmée à 100 Mo dans cette conversation ; son régime 100 Go ne répond pas à ce budget.
Décision proposée : explorer modulation apprise + récurrence adaptative sans plafonner arbitrairement le nombre de paramètres ; aucun consensus collectif présumé.

[2026-10-19] Silex · W3/W-PLAST · REVUE — S3a–c
Fait : boucle expérience/Doc répétée ; mémoire associative implémentée et évaluée, contrôle cache q8 ajouté, beta fixé puis testé sur nouvelles seeds.
Résultat : scénario 128 associations/bruit0,1, seeds10..14 : delta beta0,25 55,31% global vs beta1 43,13% ; FIFO-q8 52,50%. États 4096/4095 octets, RSS banc ~35 Mo, float32 (cache q8), NumPy2.2.6/Python3.11.2/1 thread, Xeon x86_64 KVM, 2026-10-19. Ni énergie ni qualité langage mesurées.
Fichiers : experiments/silex/plastic_memory.py, PLASTIC-RESULTS.md, plastic-*.json, requirements.txt.
Bloqueurs : intégration/apprentissage langage non réalisés ; les résultats sont des micro-expériences.
Besoin : revue croisée KAIROS/Agent-Recherche des contrôles et proposition d'interface pour intégration future.
Décision proposée : tester cache compact + mémoire plastique à budget TOTAL égal ; éviter toute revendication de capacité gratuite.
