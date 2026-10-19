# AGENTS.md — Protocole de collaboration

Ce dépôt est travaillé par **plusieurs agents en parallèle**, chacun sur sa propre branche.
Ce document définit comment éviter les collisions, comparer les résultats et converger.

---

## 1. Règle d'or

**Une branche = un chantier = un agent.** Personne ne pousse sur la branche d'un autre.
Les échanges passent par le Google Doc **« Project LLM »** (temps réel) et par ce dépôt (versionné).

---

## 2. Chantiers (workstreams)

| ID | Chantier | Périmètre | Livrable |
|---|---|---|---|
| **W1** | Architecture & modèle | Choix archi, hyperparamètres, code du modèle | Config + courbes |
| **W2** | Données & distillation | Corpus, filtrage, enseignant, génération, SFT | Dataset + recette |
| **W3** | Compression & kernels | QAT, ternaire, GGUF, bitnet.cpp, latence | Modèle quantifié + bench |
| **W4** | Système & produit | RAG, outils, grammaire, cascade, API | Démo mesurable |
| **W5** | Métrologie & énergie | RSS, RAPL/Joulescope, benchmarks, reproductibilité | Protocole + chiffres |
| **W6** | Coordination | Doc, ADR, synthèse, arbitrages | Décisions |

Un agent peut couvrir plusieurs chantiers, mais **doit déclarer lesquels** dans
`coordination/agents.md`.

---

## 3. Nommage des branches

```
agent/<nom-court>/<chantier>/<sujet>
ex. agent/marie/w3-qat-4bit
    agent/karim/w2-distillation-fr
```

La branche `main` ne reçoit que des synthèses validées (docs, configs, résultats), jamais du code
en cours de développement.

---

## 4. Format de message obligatoire (journal)

Tout agent qui termine une étape poste dans le Google Doc, onglet **« 11 · Journal des agents »**,
ET dans `coordination/journal.md`, ce bloc :

```
[YYYY-MM-DD] <agent> · <chantier W?> · <statut>
Fait      : ...
Résultat  : ... (chiffres, pas d'adjectifs)
Fichiers  : chemins dans le dépôt
Bloqueurs : ... / aucun
Besoin    : ... / rien
Décision proposée : ... / aucune
```

Statuts : `EN COURS`, `BLOQUÉ`, `REVUE`, `FAIT`, `ABANDONNÉ`.

**Règle des chiffres** : tout résultat quantitatif doit préciser
`modèle · quantification · contexte · backend · matériel · date`.
Sans ces six champs, le chiffre n'est pas comparable et sera ignoré.

---

## 5. Décisions (ADR)

Toute décision structurante est consignée dans `coordination/decisions.md` :

```
ADR-00X — <titre>
Date :
Statut : proposé | accepté | rejeté
Contexte :
Décision :
Conséquences :
Qui a validé :
```

Une décision acceptée ne se rediscute pas sans un nouvel ADR. Les désaccords se règlent par
**expérience**, pas par argument d'autorité.

---

## 6. Définition de « terminé »

Un livrable n'est `FAIT` que si :
1. il est reproductible (script + commande + seed) ;
2. il inclut une mesure de **RSS** et de **débit** ;
3. il précise la **qualité sur la tâche cible** ;
4. il est documenté dans le dépôt (pas seulement dans le chat ou le doc).

---

## 7. Ce qui est hors sujet (pour l'instant)

- Entraîner depuis zéro un modèle > 300 M.
- Tout ce qui ajoute de la mémoire à l'inférence (MoE, ensembles, multi-modèle).
- Les benchmarks généraux comme unique critère de succès.
- Les optimisations non mesurées.

---

## 8. Conflits

Si deux agents produisent des résultats contradictoires :
1. comparer les protocoles de mesure (§4, six champs) ;
2. refaire l'expérience sur la même machine ;
3. à défaut, **le résultat le plus prudent l'emporte** jusqu'à preuve du contraire.

---

## 9. Priorités actuelles

1. **E0** — mesurer le plancher de runtime (RSS d'un modèle 40 Mo).
2. **E1** — entraîner un 60 M sur 1 Md tokens et mesurer la perplexité.
3. **E4** — distillation enseignant → 75 M.
4. **E7** — RAG + grammaire sur la tâche cible.

Les chantiers W3 (ternaire) et W5 (énergie) peuvent avancer en parallèle.
