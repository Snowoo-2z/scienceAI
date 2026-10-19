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
