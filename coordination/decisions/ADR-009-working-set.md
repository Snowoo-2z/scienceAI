# ADR-009 — Le working set est la variable de conception

**Date** : 2026-10-19
**Statut** : proposé (branche `arena/ab1c7e6e-scienceai`, agent MNÉMOSYNE)
**Contexte** : le projet a jusqu'ici raisonné en « nombre de paramètres qui
tiennent dans X Mo ». Cette formulation plafonne le produit à un dense de
~120 M sous 100 Mo et à Gemma 3 270M sous 200 Mo. Or la RAM nécessaire à
l'inférence dépend de trois choses : le nombre d'octets **résidents**, le
nombre d'octets **actifs par token**, et le nombre d'octets **chargés depuis
le disque par requête**. Un modèle peut stocker beaucoup plus de paramètres
que son working set, si seuls quelques experts sont actifs et si le reste vit
sur disque.

**Décision** : adopter le **working set** (`RSS` + octets actifs + octets
chargés par requête) comme variable de conception, et ouvrir le chantier
W1-ter « ALEXANDRIE » :

- cœur de 40–60 M de paramètres, 4 bits, distillé ;
- bibliothèque de cartouches de compétence (deltas de rang faible, base
  partagée) sur disque ;
- routeur qui charge 1 à 3 cartouches par requête ;
- composition par arithmétique de tâches ;
- vérificateur et retrieval pour la rigueur et les faits.

La baseline produit (Gemma 3 270M QAT q4, 200 Mo) n'est pas modifiée. Le
chantier est une branche de recherche, pas une recommandation de déploiement.

**Conséquences** :

1. Les budgets doivent désormais rapporter quatre chiffres : RSS processus,
   page cache, octets lus sur disque par requête, taille de la bibliothèque.
2. Le critère de succès devient : à RSS égal, le modular bat-il le dense
   120 M q4 de +5 points en moyenne et +15 points sur les compétences
   spécialisées, à ≤ 2× la latence ?
3. Les expériences E-M1 (rang nécessaire de la base partagée) et E-M2
   (modular vs dense à RSS égal) sont les portes go/no-go.
4. La cible 100 Mo n'est plus interprétée comme « 100 Mo de fichier de
   poids », mais comme « 100 Mo de working set, hors stockage ».

**Qui a validé** : à valider par l'humain et par W6. En attendant, le
chantier est ouvert et les résultats sont étiquetés `MESURÉ` / `CALCULÉ` /
`HYPOTHÈSE`.

— MNÉMOSYNE
