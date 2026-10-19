# ADR-006 — Traiter deux régimes de budget, pas un seul

- **Statut :** proposé (à arbitrer par l'humain)
- **Date :** 2026-10-19
- **Auteur :** KAIROS
- **Chantier :** W6 / W1-bis

---

## Contexte

Le brief humain dit : « faire tourner un **grand LLM** dans **une centaine de
RAM** ». Deux lectures ont émergé :

1. **100 Mo** (onglets 00-11, agent `9c2ae177`) : modèle spécialisé de 60-85 M
   paramètres, distillation, RAG, grammaire. Conclusion solide : un généraliste
   sous 100 Mo n'existe pas ; un spécialisé est faisable.
2. **100 Go** (cet agent) : modèle de 100-450 B paramètres, MoE ultra-sparse,
   ternaire. Conclusion : un 317 B à 6,4 B actifs tient dans 82,6 Go et tourne à
   68 tok/s sur une machine à 3 500 $.

Les deux travaux sont rigoureux et **ne se contredisent pas** : ce sont deux
branches de la même loi de budget (`docs/HECATON-LOI-BUDGET.md`). Sous ~300 Mo,
les coûts fixes du runtime dominent ; au-dessus de ~1 Go, c'est la bande
passante. Optimiser l'un ne dit rien sur l'autre.

## Décision proposée

1. **Ne pas trancher avant mesure.** Les deux pistes coûtent < 5 000 $ chacune
   avant leur premier point de décision.
2. **Nommer explicitement le régime** dans tout chiffre publié. Un débit, un RSS
   ou un coût sans régime est ininterprétable.
3. **Priorité scientifique : le régime 100 Go.** C'est le seul qui répond
   littéralement au mot « grand », et c'est celui dont les inconnues (qualité du
   ternaire à 300 B) sont les plus intéressantes.
4. **Priorité produit : le régime 100 Mo.** C'est le seul livrable en semaines,
   et il a un chemin de mise sur le marché clair.
5. **Métrique commune : J / réponse juste.** Elle est valide dans les deux
   régimes et empêche de comparer des pommes et des Mo.

## Conséquences

- Deux jeux de fichiers, deux jeux de budgets, **un seul journal**.
- Les expériences E17/E18 (100 Go) et E0/E7 (100 Mo) sont indépendantes et
  peuvent avancer en parallèle.
- L'ADR-003 (NanoFR-75M comme référence) reste valide **pour le régime 100 Mo**.
  HECATON-300A6 est la référence du régime 100 Go. Aucune des deux ne remplace
  l'autre.
- Risque : dispersion. Mitigation : un seul agent par régime, et un point de
  décision humain à E10/E15.

## Alternatives

- **A. Forcer 100 Mo.** Rejeter HECATON. Coût : abandonne le mot « grand » du
  brief. Avantage : un seul chantier, livrable vite.
- **B. Forcer 100 Go.** Rejeter NanoFR. Coût : abandonne le seul chemin
  embarqué, et suppose que le ternaire tient la qualité (non démontré).
- **C. Attendre l'arbitrage humain avant de commencer.** Coût : jours perdus
  alors que les deux premières expériences sont peu chères.

## Références

- Onglets 00-11 du document partagé (agent `9c2ae177`)
- `docs/HECATON-100GO.md` §1 et §5
- `docs/HECATON-LOI-BUDGET.md` §3
