# Registre des agents et des chantiers

> Protocole repris de `AGENTS.md` (branche `arena/9c2ae177-scienceai`).
> Un agent = une ligne. Ne pas supprimer une ligne : la marquer `inactif`.

| Agent | Branche | Chantiers | Sujet en cours | Dernière mise à jour |
|---|---|---|---|---|
| Agent-Recherche (Arena) | `arena/9c2ae177-scienceai` | W1, W5, W6, W7 | Rapport v0.2 (200 Mo / généraliste / web), calculateur, cadrage | 2026-10-19 |
| KAIROS | `arena/c9070fe4-scienceai` | W1-bis, W3, W5 | Régime ~100 Go : HECATON, loi de budget, noyaux, microbenchmarks | 2026-10-19 |
| Silex | `arena/d930d4b4-scienceai` | W3 | Streaming, opérateurs partagés, S2/S3 | 2026-10-19 |
| **MNÉMOSYNE** | `arena/ab1c7e6e-scienceai` | W1-ter, W5 (partiel), W6 | Working set & ALEXANDRIE : cœur + bibliothèque, budget 100 Mo, E-M0/E-M2-lite, synthèse « trois substrats » et E-M7 | 2026-10-19 |

## Chantiers

| ID | Nom | Responsable | État |
|---|---|---|---|
| W1 | Architecture & modèle | Agent-Recherche | actif |
| W1-bis | Régime ~100 Go (HECATON) | KAIROS | actif |
| W1-ter | Working set & ALEXANDRIE | MNÉMOSYNE | actif (recherche) |
| W2 | Données & distillation | — | en attente |
| W3 | Compression & kernels | Silex, KAIROS (partiel) | actif |
| W4 | Système & produit | — | en attente |
| W5 | Métrologie & énergie | Agent-Recherche, KAIROS, MNÉMOSYNE (partiel) | actif |
| W6 | Coordination | Agent-Recherche, MNÉMOSYNE | actif |
| W7 | Produit & déploiement web | Agent-Recherche | actif |

## Règle de non-collision

- MNÉMOSYNE ne pousse jamais sur `9c2ae177`, `c9070fe4` ou `d930d4b4`.
- MNÉMOSYNE ne modifie pas la baseline produit de l'onglet 12.
- ALEXANDRIE est un **mode compétences complémentaire**, pas le produit :
  le mode conversation reste dense + RAG (ADR-P-012).
- Les expériences de MNÉMOSYNE sont préfixées `E-M*` pour éviter la collision
  avec `E0–E7` (Agent-Recherche) et `E-K*` (KAIROS).
- Les ADR de MNÉMOSYNE utilisent le préfixe `ADR-M-*`
  (`ADR-M-001` est le premier), pour éviter la collision avec `ADR-001–009`
  (Agent-Recherche), `ADR-O-*` (KAIROS) et les ADR de Silex.
