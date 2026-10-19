# Journal des agents — branche MNÉMOSYNE

> Journal versionné de la branche `arena/ab1c7e6e-scienceai`. L'historique
> complet et en temps réel est dans l'onglet 11 du document partagé ; les
> journaux des autres agents sont dans leurs branches respectives.
> Format : voir `AGENTS.md` §4. Ajouts en fin de fichier.

---

```
[2026-10-19] MNÉMOSYNE · W1-ter/W5/W6 · EN COURS
Fait      : Lecture du document partagé (onglets 00, 01, 11, 12 et onglet
            KAIROS), des branches 9c2ae177 / c9070fe4 / d930d4b4, de
            AGENTS.md, des ADR-001 à 008, de docs/ARCHITECTURE-200MB-WEB.md,
            de HECATON.md et de docs/00-CADRAGE-KAIROS.md.
            Ouverture du chantier W1-ter « working set & ALEXANDRIE ».
            Calculateur de budget tools/working_set.py (stdlib).
            Microbenchmark de chargement de cartouches tools/wsprobe.c.
Résultat  : CALCULÉ — ALEX 60 M + bibliothèque de 10 Md de paramètres
            ternaires : RSS 69,2 Mo, poids actifs 34,2 Mo, bibliothèque
            2 533,8 Mo sur disque, 161,3 tok/s, 19 J/réponse de 200 tokens
            (GOPS=20, plateforme 15 W, chargement 2 Mo à 1 Go/s).
            Dense 120 M q4 : RSS 99,5 Mo, 67,5 Mo actifs, 83,3 tok/s,
            36 J/réponse. Gemma 3 270M QAT q4 8k : 176,5 Mo, 74,6 tok/s,
            40 J/réponse.
            MESURÉ (bac à sable, 2 vCPU Xeon 2,60 GHz, 4 Go, ext4,
            2026-10-19) — chargement médian d'une cartouche, fichier 2 Go,
            pread bufferisé + fadvise/drop_caches : 1 Mo 1,01 ms ; 2 Mo
            1,86 ms ; 4 Mo 3,07 ms ; 8 Mo 5,75 ms. En O_DIRECT : 1 Mo
            0,34 ms ; 2 Mo 0,61 ms ; 8 Mo 2,45 ms. Débit aléatoire 0,26 à
            1,4 Go/s ; séquentiel 0,8 à 3,4 Go/s. p99 observés ~100 ms sur
            VM partagée.
            HYPOTHÈSE — la qualité d'une bibliothèque de 10 Md de paramètres
            n'est pas démontrée ; c'est l'objet de E-M1/E-M2.
Fichiers  : docs/research/mnemosyne-working-set.md
            coordination/decisions/ADR-M-001-working-set.md
            coordination/agents.md, coordination/journal.md
            tools/working_set.py, tools/wsprobe.c
            results/working-set-configs.txt, results/working-set-law.txt,
            results/working-set-configs.csv,
            results/wsprobe-sandbox.txt, results/wsprobe-sandbox-2g.txt,
            results/wsprobe-sandbox-direct.txt
Bloqueurs : aucun numpy ni torch dans le bac à sable ; pas de GPU ; 2 vCPU.
            E-M1/E-M2 (qualité, rang, modular vs dense) exigent un GPU loué
            et des données de compétence. L'unité exacte des 100 Mo
            (RSS processus ou RAM totale, page cache inclus) n'est pas
            tranchée. La taille de disque acceptée sur la cible n'est pas
            connue.
Besoin    : (a) confirmation humaine : 100 Mo = RSS processus hors page
            cache, et disque de 1–10 Go acceptable ? (b) 20 compétences ×
            200 exemples + labels de routage (W2) ; (c) accès GPU pour
            E-M1/E-M2 ; (d) protocole S3 de Silex ; (e) p99 de chargement
            sur le matériel cible (KAIROS/W5).
Décision proposée : ADR-M-001 — le working set est la variable de conception ;
            ouvrir W1-ter ALEXANDRIE ; ne pas modifier la baseline produit
            Gemma 3 270M (200 Mo).
```

## Message aux autres agents

**@KAIROS** — ta loi `tok/s = BW/(B·ρ)` et ton seuil `GOPS ≥ 16·BW/bits`
sont la base de la mienne ; je les étends à une hiérarchie RAM/disque avec
chargement par requête. Ton régime ~100 Go est la borne haute de la même
loi : chez toi tout est en RAM et le noyau est le goulet ; chez moi le cœur
est petit, le disque existe, et le routage est le goulet. Peux-tu mesurer le
p99 (pas la moyenne) d'un `pread` aléatoire de 1–8 Mo sur ta machine cible ?
C'est E-M0 côté matériel. Ton ADR-007 est le bon cadre ; je propose de le
préciser : le rendement marginal se mesure par octet de **working set**, pas
par octet de fichier.

**@Silex** — ton opérateur `W_i = U diag(c_i) V` est le format de cartouche
idéal : la base `U, V` reste en RAM, la diagonale `c_i` est la compétence,
et l'addition de deux diagonales est une composition. Ton résultat négatif
(« plus d'opérateurs ≠ plus de paramètres indépendants ») est le bon
garde-fou. Proposition : co-écrire S3 comme E-M1/E-M2 — tu fournis le
prototype et les tests d'équivalence, j'apporte le budget, le routeur et le
jeu de compétences. On publie les deux résultats, positifs ou négatifs.

**@Agent-Recherche** — je ne touche pas à l'onglet 12 ni au produit Gemma
270M. Ta baseline est le point de comparaison à 200 Mo. J'ai besoin de la
config exacte du GGUF Q4_0, du RSS mesuré en E0, du jeu d'évaluation de la
tâche cible et du budget de tokens du QLoRA. Si E-M2 échoue, je le dirai ;
si E-M2 réussit, on aura une voie vers un généraliste à 100 Mo qui complète
ton produit à 200 Mo.

**@W2/W4/W5/W7** — W2 : 20 compétences × 1–5 M tokens + labels de routage,
c'est le vrai goulot. W4 : routeur, index, vérificateur, grammaire. W5 :
comptabilité RSS/page cache/disque et `J/réponse juste`. W7 : chargeur de
cartouches dans le serveur, cache, supervision.

— MNÉMOSYNE

---

```
[2026-10-19] MNÉMOSYNE · W1-ter · EN COURS
Fait      : E-M2-lite — jouet de capacité et de composition de la base
            partagée W_k = U diag(c_k) V. d=32, 5 graines, rangs 1 à 32.
            Ce n'est pas un LLM : c'est un test de la paramétrisation.
Résultat  : MESURÉ (Python stdlib, 2026-10-19) —
            matrice dense aléatoire : erreur 0,996 à r=8, 0,982 à r=32
            (une base fixe ne représente pas une compétence arbitraire) ;
            compétence dans la base : 0,000 (fit exact) ;
            compétence dans la base + 10 % de bruit : 0,287 à r=8,
            0,089 à r=32 ;
            compétence dans une autre base : ~0,997 à 1,000 (pas de
            transfert) ;
            composition c_A + c_B vs meilleur ajustement direct :
            écart < 0,001 (r=8 : 0,135 vs 0,135 ; r=32 : 0,074 vs 0,073) ;
            stockage : 672 paramètres partagés vs 20 480 denses pour 20
            compétences à r=8 (32,8× plus compact ; 244× à r=1, 7,6× à r=32).
Fichiers  : experiments/mnemosyne/toy_shared_basis.py
            results/toy-shared-basis.txt, results/toy-shared-basis.json
            docs/research/mnemosyne-working-set.md §6.1bis
Bloqueurs : jouet synthétique uniquement ; aucune mesure sur de vrais poids
            ni sur un LLM. Le résultat négatif sur la base partagée doit
            être confirmé par E-M1 sur des poids réels.
Besoin    : @Silex — relire le protocole et dire si le mésappariement de
            base correspond à ce que tu observes dans S2/S3. @KAIROS —
            la composition par addition te paraît-elle compatible avec ta
            loi de budget ?
Décision proposée : par défaut, cartouches LoRA par compétence (Mode B,
            U_k et V_k propres, 0,1–0,3 Mo) ; la base partagée (Mode A,
            c_k seul) reste un pari de compression extrême, seulement si
            E-M1 montre que les compétences réelles sont alignées.
```

---

```
[2026-10-19] MNÉMOSYNE · W1-ter/W6 · CORRECTION
Fait      : Lecture du commit 791ac33 de la branche 9c2 (« Corrige la v0.2
            avec les tailles GGUF mesurées ») et du Programme II de KAIROS
            (onglet 13, ADR-O-007, E-Ω0/Zipf).
Résultat  : Les tailles GGUF mesurées invalident l'estimation initiale de
            Gemma 3 270M : 253 Mo de fichier, 286–296 Mo de RSS, hors
            budget 200 Mo. La baseline produit devient MobileLLM-R1-140M
            Q4_K_M (132 Mo, 170–182 Mo de RSS) ou Gemma 270M après élagage
            du vocabulaire à 32 k (106 Mo calculé, 136–141 Mo de RSS).
            SmolLM2-135M Q4_K_M : 105 Mo, 155–178 Mo de RSS.
            Notre note et tools/working_set.py sont mis à jour : champ
            `file_mb` (taille mesurée prioritaire sur bits/poids), presets
            mobilellm-140m-q4km, smollm2-135m-q4km, gemma-270m-q4km,
            gemma-270m-vocab32k. L'ancienne ligne « Gemma 270M à 176,5 Mo »
            est retirée.
            ADR-009 étant pris par « la taille du vocabulaire est le levier
            mémoire n° 1 », notre ADR devient ADR-M-001 (préfixe M pour
            MNÉMOSYNE ; KAIROS propose ADR-P-* produit et ADR-O-* recherche Ω).
Fichiers  : docs/research/mnemosyne-working-set.md §2.3 et §8
            coordination/decisions/ADR-M-001-working-set.md
            tools/working_set.py, results/working-set-configs.*
            README.md, coordination/agents.md
Bloqueurs : aucun.
Besoin    : @Agent-Recherche — confirmer que la baseline MobileLLM-R1-140M
            est bien celle de l'onglet 12 après correction, et donner le RSS
            mesuré en E0. @KAIROS — ta convention ADR-P-*/ADR-O-* est
            adoptée ; j'ajoute ADR-M-* pour la branche working set.
Décision proposée : entériner ADR-M-001 et la correction de baseline ;
            toute comparaison à 100 Mo utilise désormais MobileLLM-R1-140M
            ou le dense 120 M, plus Gemma 270M.
```

---

```
[2026-10-19] MNÉMOSYNE · W1-ter/W6 · EN COURS
Fait      : Lecture de l'onglet 13 (Programme II de KAIROS), des résultats
            S3a–c de Silex (PLASTIC-RESULTS.md) et du commit 791ac33 de 9c2
            (baseline GGUF).
Résultat  : Synthèse « trois substrats, une même loi » : poids denses,
            compétences-code, mémoire épisodique, retrieval, cartouches LoRA.
            Le résultat qui décide : à 1 Go/s effectif, relire un dense 270 M
            coûte 135 ms/token, un dense 7 B 3,5 s/token ; charger une
            compétence-code de 5,5 ko coûte 5 µs/requête, une cartouche LoRA
            de 2 Mo 2,1 ms. Le chargement modulaire est quatre ordres de
            grandeur sous le streaming dense.
            À 20 Mo de cache et α=1, 3 723 compétences-code couvrent 89,9 %
            des requêtes ; 10 cartouches LoRA couvrent 29,9 %. Le code est
            ~3× plus compact que le poids pour la même couverture, mais
            seulement sur le vérifiable. À α=0,7, les mêmes 3 723 skills ne
            couvrent que 72,4 % : α et k restent les deux inconnues qui
            peuvent tuer le programme.
Fichiers  : docs/research/mnemosyne-working-set.md §11 (synthèse, E-M7) ;
            tools/substrate_budget.py ; results/substrate-budget.txt.
Bloqueurs : aucune mesure de qualité ; α et k inconnus (pas de logs) ;
            aucune intégration llama.cpp.
Besoin    : @KAIROS — α et k sur des logs réels, et confirmation que la loi
            de couverture H_n/H_N est la bonne ; @Silex — rappel de la
            mémoire plastique à 1 000/10 000 associations, et courbe
            rang-capacité de la base partagée sur de vrais poids ;
            @Agent-Recherche — le jeu de 500 requêtes de la tâche cible et
            le RSS mesuré de MobileLLM-R1-140M en E0.
Décision proposée : ADR-M-001 (working set comme variable de conception) ;
            E-M7 « quatre substrats à 100 Mo » (dense 120 M vs cœur 60 M +
            skills / LoRA / mémoire / mix) ; la baseline produit reste
            MobileLLM-R1-140M.
```

---

```
[2026-10-19] MNÉMOSYNE · W1-ter/W5 · EN COURS
Fait      : Réponse à Agent-Recherche (onglet 13) ; test page cache (E-M0b) ;
            mise à jour du périmètre.
Résultat  : E-M0b MESURÉ (bac à sable, 1,91 GiB, 8 Mio, cgroup /user) :
            lecture bufferisée → memory.current 55,8 → 2 019,5 Mio,
            VmRSS 34,7 Mio, 836 Mio/s ; avec posix_fadvise(DONTNEED) →
            memory.current 61,8 Mio, VmRSS 34,8 Mio, 625 Mio/s.
            Le page cache gonfle le cgroup de la taille du fichier alors que
            le RSS reste à 35 Mo ; DONTNEED le ramène à 62 Mo pour −25 % de
            débit. Un plafond MemoryMax=100M évincerait le page cache, voire
            le cœur : E-M5 doit mesurer les deux.
            Périmètre accepté : le produit est la conversation ouverte ;
            ALEXANDRIE est un mode compétences complémentaire (ADR-P-012),
            pas le produit. La baseline conversation reste dense + RAG.
Fichiers  : tools/pagecache_probe.py ; results/pagecache-probe.txt ;
            docs/research/mnemosyne-working-set.md §0, §6.1ter, §8.3, §9 ;
            README.md ; coordination/decisions/ADR-M-001-working-set.md ;
            coordination/agents.md.
Bloqueurs : aucune mesure de qualité ; α/k inconnus ; pas de jeu d'évaluation ;
            pas de GPU.
Besoin    : @Agent-Recherche — jeu de 200–500 requêtes, logs anonymisés,
            cible matérielle, accord GPU. @KAIROS — α et k sur logs ; noyau
            de chargement p99. @Silex — rappel plastique à 1k/10k
            associations ; rang-capacité sur vrais poids.
Décision proposée : ADR-M-001 accepté (mode compétences) ; E-M5 inclut le
            test memory.current sous MemoryMax=200M.
```


---

```
[2026-10-19] MNÉMOSYNE · W4/W1-ter · FAIT (démo exécutable)
Fait      : Une démo qui tourne, que l'humain peut essayer dans un navigateur.
            demo/server.py + demo/index.html + tools/demo_assistant.py.
            12 compétences vérifiées, 1 880 passages indexés, 0 paramètre appris,
            aucun réseau, aucun GPU.
Résultat  : MESURÉ (Xeon 2,60 GHz, 2 vCPU, 4 Go, Python 3.11.2, 2026-10-19) :
            index construit en 34 ms ; 13 ms par question (12,6 ms sur 20) ;
            VmRSS du serveur 34,8 Mo ; memory.current 61,8 Mo (page cache inclus).
            Jeu de 20 questions : 16/20 au premier passage ; 20/20 après ajout
            de 4 compétences — MAIS le jeu est écrit par moi et je l'ai vu,
            donc 20/20 ne vaut rien comme généralisation. 8/8 refus sur les
            questions hors corpus, refus trivial (aucune skill ne matche).
            Exemple : « Pourquoi Gemma 3 270M est-il hors budget ? » → « 253 Mo
            en Q4_K_M, 286–296 Mo de RSS, vocabulaire de 262 144 tokens »,
            source journal.md.
Fichiers  : demo/{server.py,index.html,README.md,eval-questions.json} ;
            tools/{demo_assistant.py,eval_common.py} ;
            results/{eval-common.txt,demo-assistant.txt,demo-probe.txt} ;
            coordination/dashboard.md (tableau de bord commun) ;
            onglet 14 « Discussion — atelier des agents » (t.vsnzy7y4iows).
Bloqueurs : le jeu d'évaluation est de moi → 20/20 non défendable. Il faut un
            jeu écrit par un autre agent ou par l'humain, sans que je le voie.
Besoin    : @Agent-Recherche — écrire 20 questions que je n'ai pas vues, sur
            la tâche cible, avec la réponse attendue. @Silex, @KAIROS —
            remplir une ligne du tableau de bord avec VOS chiffres, aux mêmes
            colonnes (VmRSS, memory.current, octets lus/requête, qualité).
Décision proposée : adopter le tableau de bord commun comme seul format de
            comparaison entre bras. La démo F est le seul bras entièrement
            mesuré à ce jour.
```
