# Journal — KAIROS

Format imposé par `AGENTS.md` §3.

---

## [2026-10-19] KAIROS · W1-bis · en cours

**Fait :**
- Lecture du document partagé : 12 onglets, cadrage 100 Mo par l'agent
  `9c2ae177`, protocole W1-W6 et ADR-001→005.
- Vérification du dépôt : `main` = `173b51b`, aucune autre branche distante.
  La branche `arena/9c2ae177-scienceai` citée dans le doc n'est pas poussée.
- Construction de l'outillage : `tools/hecaton_budget.py` (calculateur
  RAM/débit/énergie/entraînement, stdlib), `tools/bwprobe.c` (microbenchmark).
- Architecture HECATON-300A6 chiffrée : 317,4 B de paramètres, 6,42 B actifs,
  82,6 Go de RSS à 32 k, 2,22 Go lus/token, Λ = 37,3.
- Microbenchmark mémoire exécuté (VM 2 vCPU) : plateau de bande passante
  atteint dès 64 Ko ; dispersion des experts = **3 %** d'écart → idée
  « expert clustering » abandonnée.
- Loi de budget unifiée rédigée : `tok/s = BW / (B·ρ)`.

**Résultat :**
- Régime ~100 Go viable sur le papier : 68 tok/s (252 avec spéculation) sur
  Ryzen AI Max+ 395 128 Go, 0,88 J/token, 276 J/réponse juste.
- Entraînement de réparation (100 Md tokens) : 2 700 H100-heures ≈ 6 800 $.
- Résultat négatif : la localité des experts n'est pas un problème ; celle des
  petits tenseurs (< 4 Ko) l'est.

**Fichiers :**
`docs/HECATON-100GO.md`, `docs/HECATON-LOI-BUDGET.md`,
`docs/MICROBENCH-MEMOIRE.md`, `docs/REFERENCES.md`, `AGENTS.md`,
`coordination/decisions/ADR-006-double-regime.md`,
`tools/hecaton_budget.py`, `tools/bwprobe.c`, `configs/*.json`,
`results/*.md`, `figures/*.svg`.

**Bloqueurs :**
- Aucune machine cible disponible : tous les chiffres de débit sont des
  plafonds roofline, pas des mesures (sauf le microbenchmark).
- Le régime 100 Go dépend d'un fait non établi : la qualité du ternaire au-delà
  de 3 B de paramètres.
- Branche de l'agent `9c2ae177` absente du remote : impossible de lire son
  rapport complet ni de vérifier ses fichiers.

**Besoin :**
- Arbitrage humain sur l'unité (100 Mo vs 100 Go) — ADR-006.
- Accès à une machine 96-128 Go (Strix Halo, M4 Max) pour E10 et E15.
- Un accord sur la métrique commune J/réponse juste (W5).

**Décision proposée :**
ADR-006 : traiter les deux régimes comme deux branches de la même loi ; garder
`9c2ae177` sur 100 Mo, ouvrir W1-bis sur 100 Go ; métrique commune J/réponse.

---

## [2026-10-19] KAIROS · W5 · terminé (mesure locale)

**Fait :** `bwprobe` v2, 9 tailles de bloc (64 o → 4 Mo), tas de 1 Go,
4 experts × 4 Mo dispersés vs contigus.

**Résultat :** 20 % de la bande passante à 64 o, 62 % à 1 Ko, 84 % à 4 Ko,
99 % à 64 Ko. Gather dispersé vs contigu : 3 % d'écart.

**Fichiers :** `results/bwprobe-sandbox.md`, `docs/MICROBENCH-MEMOIRE.md`.

**Bloqueurs :** VM partagée 2 vCPU ; valeurs absolues non transférables.

**Besoin :** rejouer sur le matériel cible (E15). 1 jour.

**Décision proposée :** abandonner l'« expert clustering » ; rediriger l'effort
de localité vers le KV, le routeur et la tête de sortie.
