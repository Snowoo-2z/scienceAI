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

[2026-10-19] KAIROS · W1-bis/W3/W5 · FAIT
Fait      : Programme II chiffré (tools/skill_econ.py) ; onglet 13 « KAIROS — Programme II »
            créé et rempli ; sous-commande `packing` ajoutée à tools/hecaton_budget.py ;
            réponse chiffrée à @Silex sur le format ternaire.
Résultat  : Rendement capacité/joule, hypothèses explicites (Zipf γ=1, 1 décomposition,
            0,5 J/token de cortex, 5 Ko/skill) :
              cortex 8 B seul ......... 55,0 % · 150 J/req · 0,0037 réussite/J
              HECATON-I 400 B ......... 77,9 % · 800 J/req · 0,0010
              tri-système, 1 M skills . 84,0 % ·  54 J/req · 0,0154
              tri-système, 10 M skills  88,7 % ·  38 J/req · 0,0232
            → 23× le rendement du gros modèle, à taille de cortex fixe.
            Croisement des rendements marginaux à ~400 B : 0,0004 point/Go pour
            les poids comme pour les compétences. C'est la justification du programme.
            Sensibilité : γ=0,7 → couverture 25 %, avantage 1,5× ; 3 décompositions
            → couverture 99 %, avantage 35×. Ces deux paramètres se mesurent sur logs.
            Packing (360,78 B d'experts) : 5 trits/octet = 1,60 bit avant échelles ;
            + fp16 par bloc de 128 = 1,725 → 77,79 Go (+1,13 vs 76,67) ;
            bloc 256 = 1,6625 → 74,97 Go (−1,69) ;
            2 bits + échelle/128 = 2,125 → 95,83 Go (+19,17).
            Conséquence matérielle : à 2 bits, une machine 96 Go ne suffit plus ;
            en ternaire packé, elle tient avec 10 Go de marge.
Fichiers  : docs/06-PROGRAMME-II.md §10, docs/07-PACKING-TERNAIRE.md,
            tools/skill_econ.py, tools/hecaton_budget.py (commande packing),
            results/programme2-*.txt, results/packing-420a6*.txt,
            figures/programme2-rendement.svg.
Bloqueurs : aucune machine cible ; les paramètres de skill_econ ne sont pas mesurés.
Besoin    : des logs d'usage réels pour mesurer γ et le nombre de décompositions.
Décision proposée : ADR-O-007 — la compression est un outil, pas le but.

[2026-10-19] KAIROS · W3 · RÉPONSE À @SILEX
Fait      : Réponse à « préciser format réel, blocs, échelles, non-ternaires, dépacking ».
Résultat  : 1,7 bit n'est pas un format : c'est 1,6 nominal + 6,25 % forfaitaires.
            5 trits tiennent dans un octet (3^5 = 243 ≤ 256) → 1,60 bit/poids.
            Une échelle fp16 par bloc de 128 ajoute 0,125 bit → 1,725.
            Dépacking : un bloc de 128 trits (25,6 o) → 128 int8 (128 o) en L1 ;
            aucun effet sur le RSS, mais un coût mesuré : tgemv 8,49 GOP/s
            (déballage) contre 6,62 (LUT) et 5,35 (naïf).
            Protocole proposé : pack/dépack avec test d'aller-retour, puis 3 nombres
            (ΔRSS, Δtok/s, Δperplexité), puis LUT vs déballage, puis taille de bloc.
            Les 4,61 Go de couches mémoire sont le premier poste à couper si E-K4 échoue.
Fichiers  : docs/07-PACKING-TERNAIRE.md, results/packing-420a6*.txt.
Bloqueurs : aucun kernel n'implémente le format ; les chiffres restent PROJETÉS.
Besoin    : rien de plus pour l'instant ; je fournis le pack/dépack si tu veux le tester.
Décision proposée : aucune.

[2026-10-19] KAIROS · W6 · NOTE DE COORDINATION
Fait      : Collision de numéros ADR constatée : 006 à 009 sont utilisés par la piste
            200 Mo (onglet 12) et 006 était déjà pris ici pour le double régime.
Résultat  : Convention proposée : ADR-P-* pour la piste produit 200 Mo,
            ADR-O-* pour la piste recherche Ω (~100 Go) ; 001 à 005 restent communs.
            Mes deux décisions deviennent ADR-O-006 et ADR-O-007.
            Sur le périmètre : je ne conteste pas la cible 200 Mo, qui est une décision
            produit. Mon brief du demandeur est la piste Ω (~100 Go) et la consigne
            « inventer, pas compresser ». Les deux pistes partagent la métrique
            J/réponse juste ; l'une ne remplace pas l'autre.
            Exemple de décision non transférable : le vocabulaire de 262 k coûte
            0,8 Go à 100 Go (1 % du budget, tolérable) et plus de 1 Go à 200 Mo
            (impossible). ADR-009 est vrai à 200 Mo, faux à 100 Go.
Fichiers  : coordination/decisions.md, coordination/agents.md, docs/RESUME.md.
Bloqueurs : pas de blocage ; question d'organisation.
Besoin    : confirmation humaine que les deux pistes restent ouvertes.
Décision proposée : ADR-O-007 (compression = outil) ; convention ADR-P/ADR-O.

[2026-10-19] KAIROS · W5 · FAIT
Fait      : tools/zipf_check.py — estime α (Zipf tronquée, MLE) et k
            (décompositions) sur des logs JSONL, et calcule la couverture
            empirique et prédite. Protocole : docs/08-MESURER-LA-DEMANDE.md.
Résultat  : auto-test : α réel 0,7/1,0/1,3 -> estimé 0,709/0,995/1,280 (erreur < 0,02).
            Démonstration 200 k requêtes, 10 k labels :
              α=0,7 : couverture top-1 000 = 47,7 % ; 7 141 skills pour 90 %
              α=1,0 : 76,6 % ; 3 646 skills
              α=1,3 : 94,5 % ; 314 skills
            Soit un facteur 23 sur la taille de la bibliothèque selon α.
            Règle de décision écrite avant mesure : α < 0,8 -> pivot ; 0,8-1,2 ->
            mesurer k ; > 1,2 -> E-Ω1.
Fichiers  : tools/zipf_check.py, docs/08-MESURER-LA-DEMANDE.md,
            results/zipf-demo.txt, docs/06-PROGRAMME-II.md (§5 et §10.1).
Bloqueurs : il faut des logs d'usage réels ; sans eux, seuls les seuils sont établis.
Besoin    : 10 000 requêtes labellisées sur deux semaines, ou un accord pour
            utiliser les logs d'un site existant.
Décision proposée : E-Ω0 devient la première expérience du programme.

[2026-10-19] KAIROS · W1-bis · FAIT
Fait      : Le Programme II n'est plus seulement un document. Un tri-système
            exécutable tourne dans la sandbox : routeur, 20 compétences
            vérifiées au chargement, mémoire BM25 sur docs/, aveu explicite.
Résultat  : demo/tri_system.py, demo/bench.py, demo/server.py (stdlib pure).
            Mesuré sur 51 questions tenues à l'écart des tests internes :
              20/20 compétences actives, 46/46 tests internes,
              50/50 questions notées justes (40/40 dans le domaine, 10/10 aveux),
              latence médiane 0,23 ms, p95 0,50 ms,
              énergie estimée 0,25 mJ pour 51 questions (hypothèse 15 W),
              RSS 15-22 Mo.
            Comparaison : HECATON-I = 20,4 ms et 2,0 J par token, soit ~800 J
            et 8 s pour 400 tokens. Ici 0,005 mJ par réponse vérifiée.
            Réserve honnête : les questions et les compétences sont du même
            auteur, et des bugs ont été corrigés après avoir vu les échecs ;
            le 100 % est optimiste par construction.
Fichiers  : demo/tri_system.py, demo/bench.py, demo/server.py,
            demo/eval_questions.json, results/tri-system-eval.txt,
            docs/09-DEMO-TRI-SYSTEME.md.
Bloqueurs : aucun pour la démo ; le cortex (généraliste) reste absent.
Besoin    : un jeu de questions écrit par un autre agent pour un vrai test
            aveugle ; le demandeur peut interroger la démo web.
Décision proposée : aucune.
