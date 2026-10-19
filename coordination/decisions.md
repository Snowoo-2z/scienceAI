# Journal des décisions (ADR)

> **Convention de numérotation (2026-10-19).** Deux pistes coexistent et ont
> des ADR distincts : `ADR-P-*` pour la piste **produit** (cible 200 Mo de
> RSS, @9c2ae177) et `ADR-O-*` pour la piste **recherche Ω** (~100 Go,
> KAIROS). Les ADR-001 à 005 sont communs aux deux. Un numéro nu ne doit
> plus être attribué.

---

## ADR-001 — Interpréter la cible mémoire comme le RSS du processus d'inférence

**Date** : 2026-10-19
**Statut** : accepté (par défaut, à confirmer par l'utilisateur)
**Contexte** : « ~100 Mo de RAM » peut désigner le fichier de poids, la mémoire du modèle, ou le RSS
du processus. Les trois diffèrent d'un facteur 1,5 à 2.
**Décision** : la cible est le **RSS au pic** du processus d'inférence, incluant poids, KV cache,
runtime et buffers de calcul, hors système d'exploitation.
**Conséquences** : un modèle de 135 M en Q4 (110 Mo de poids) est **hors budget**. La frontière se
situe autour de 85 M paramètres en 4-bit.

---

## ADR-002 — Vocabulaire de 32 768 tokens, embeddings partagés

**Date** : 2026-10-19
**Statut** : proposé
**Contexte** : la table d'embeddings consomme `vocab × d_model` paramètres. À 100 Mo, un vocabulaire
de 128 k ou 256 k consomme 40 à 85 Mo à lui seul.
**Décision** : vocabulaire 32 768, embeddings entrée/sortie partagés (tied).
**Conséquences** : économie d'environ 30 Mo par rapport à un vocabulaire de 128 k. Perte attendue
sur le code et les langues non latines. À valider par l'expérience E2.

---

## ADR-003 — Cible d'architecture : 60–85 M paramètres, 4-bit QAT

**Date** : 2026-10-19
**Statut** : proposé
**Contexte** : à 4,5 bits/poids, la frontière des 100 Mo avec 2 k de contexte est ~85 M paramètres.
**Décision** : architecture de référence NanoFR-75M (20 couches, d=512, 8/2 têtes, FFN 1 408,
vocab 32 k, contexte 2 048, KV q8).
**Conséquences** : RSS estimé 78 Mo, marge de 22 Mo. Le ternaire 200 M reste une piste exploratoire.

---

## ADR-004 — L'énergie est mesurée par réponse utile, pas par token

**Date** : 2026-10-19
**Statut** : accepté
**Contexte** : à petite échelle, `E ≈ P_plateforme / débit`. Un modèle plus petit mais plus lent
peut consommer davantage.
**Décision** : toute comparaison énergétique rapporte `J/réponse juste`, le débit et la puissance de
la plateforme.
**Conséquences** : les optimisations de latence sont aussi des optimisations énergétiques.

---

## ADR-005 — Ne pas utiliser de MoE ni de sparsité pour la cible 100 Mo

**Date** : 2026-10-19
**Statut** : accepté
**Contexte** : en décodage batch=1, le débit CPU est limité par la bande passante mémoire. Un MoE
réduit les FLOPs mais augmente l'empreinte.
**Décision** : architecture dense uniquement, sauf démonstration contraire mesurée.

---

## ADR-O-006 — Traiter deux régimes de budget (100 Mo et ~100 Go), pas un seul

**Date** : 2026-10-19
**Statut** : proposé (à arbitrer par l'humain)
**Auteur** : KAIROS
**Contexte** : Le brief dit « un grand LLM dans une centaine de Go de RAM ».
Le présent document, lui, s'est cadré sur **100 Mo** (ADR-001), soit un
facteur 1 000 d'écart. Les deux lectures sont défendables, mais elles
produisent des architectures, des budgets et des expériences différentes.
Les travaux des deux branches sont rigoureux et ne se contredisent pas :
ce sont deux points de la même courbe (`tok/s = BW / (ρ·B + KV)`).
**Décision** :
1. Ne pas trancher avant mesure : les deux premières expériences coûtent
   moins de 5 000 $ chacune.
2. Étiqueter chaque chiffre avec son régime. Un débit ou un RSS sans budget
   mémoire est ininterprétable.
3. Maintenir deux documents : `docs/RECHERCHE-LLM-100MB.md` (100 Mo) et
   `docs/00..05` + `HECATON.md` (100 Go), avec renvoi croisé.
4. Métrique commune : **J / réponse juste**, valide dans les deux régimes.
5. Priorité scientifique au régime 100 Go (c'est le seul qui réponde au mot
   « grand ») ; priorité produit au régime 100 Mo (seul livrable en semaines).
**Conséquences** : deux jeux de fichiers et de budgets, un seul journal.
Les expériences E0/E1/E7 (100 Mo) et E-K1/E-K2/E-K8 (100 Go) sont
indépendantes et peuvent avancer en parallèle. Risque : dispersion.
Mitigation : un agent par régime, et un point de décision humain après E-K8.
**Qui a validé** : en attente (KAIROS propose, l'humain tranche).

---

## ADR-O-007 — La compression devient un outil de mesure, pas le but du programme

**Date** : 2026-10-19
**Statut** : proposé (piste Ω)
**Auteur** : KAIROS
**Contexte** : HECATON-I démontre qu'un transformer de 317 B tient dans
82,6 Go à 2,01 bit, avec 49 tok/s sur un noyau à 800 GOP/s. C'est un
résultat de compression, pas une invention : le modèle continue de payer
connaissance, compétence et raisonnement aux mêmes poids, à chaque token
(2,05 Go lus par token). Le demandeur a demandé explicitement autre chose :
« trouver un moyen de créer, pas limiter la taille du modèle ».
**Décision** :
1. HECATON-I est **gelé comme baseline** reproductible. On n'y investit plus
   d'effort d'optimisation, sauf pour les mesures nécessaires aux comparaisons.
2. Le programme actif est le **tri-système** de `docs/06-PROGRAMME-II.md` :
   cortex 15 Go + bibliothèque de compétences 55 Go + mémoire plastique 25 Go
   + vérificateur 3 Go.
3. Le packing, la quantification et la sparsité ne sont retenus que s'ils
   **augmentent la capacité à joules égaux**. Un gain de 0,1 bit/poids n'est
   pas un résultat ; un facteur sur les octets lus par token en est un.
4. Toute affirmation de capacité doit être mesurée par E-Ω5 (élasticité
   capacité/Go par substrat) ou par E-Ω1 (skills contre RAG à joules égaux).
**Conséquences** : les chiffres de HECATON-I restent cités comme référence
(baseline), jamais comme livrable. Les efforts W-SKILL, W-PLAST et W-DEPTH
deviennent prioritaires sur W3 (kernels), qui passe en soutien.
**Qui a validé** : en attente (KAIROS propose, l'humain tranche).
