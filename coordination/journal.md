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

---

[2026-10-19] KAIROS · W1-bis/W3/W5 · EN COURS
Fait      : Régime ~100 Go traité en parallèle du régime 100 Mo.
            Calculateur de budget (stdlib) + 2 microbenchmarks C + 6 configs.
Résultat  : HECATON-420A6 = 372 B de paramètres, 6,0 B actifs, 84,9 Go de RSS,
            2,05 Go lus/token, 49 tok/s sur Strix Halo (noyau 800 GOP/s).
            HECATON-215A6 = 213 B, 49,5 Go, même débit (+75 % de capacité, 0 tok/s).
            SEUIL CLÉ : à 1,58 bit, on est calcul-borné tant que
            GOPS < 16*BW/bits. Mesuré : 8,5 GOP/s sur 2 vCPU (noyau C naïf).
            NÉGATIF : l'expert clustering ne sert à rien (gather dispersé =
            105-107 % du séquentiel dès 256 Ko) ; trier les indices n'aide pas.
            NÉGATIF : sous 4 Ko de bloc, on perd 20 à 65 % de bande passante.
Fichiers  : HECATON.md, docs/00..05, docs/RESUME.md, docs/REFERENCES.md,
            tools/hecaton_budget.py, tools/bwprobe.c, tools/tgemv.c,
            configs/*.json, results/*.txt, figures/*.svg
Bloqueurs : ambiguïté Go/Mo non tranchée ; aucun matériel cible disponible ;
            pas de GPU pour E-K2/E-K8 ; la branche 9c2ae177 a maintenant un
            rapport complet mais un ADR-001 fixe la cible à 100 Mo.
Besoin    : arbitrage humain sur l'unité ; accès à une machine 96-128 Go
            pour E-K1/E-K3/E-K7 ; un volontaire W2 pour les données.
Décision proposée : ADR-006 — traiter les deux régimes comme deux points de la
            même loi, avec J/réponse juste comme métrique commune.
