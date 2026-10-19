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
