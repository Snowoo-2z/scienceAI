# Silex — RAM minimale : compression ou déplacement du coût ?

2026-10-19 · W3 · **REVUE / analyse théorique, aucun benchmark LLM exécuté**.

## Contexte et limites

Lecture des onglets accueil, journal et leviers du document partagé ; lecture
à distance d'AGENTS.md et ADR-001 à ADR-005 au commit
`26ae62d8f5af3901dff662ab0c22de35dfb2dd53`.
Cette note ne remplace pas ces décisions. Le dense reste la référence ; pas de
MoE, d'entraînement ni de changement de vocabulaire engagé.
La cible provisoire du document est 100 Mo RSS, hors OS. L'utilisateur doit
encore confirmer l'unité, le matériel, la qualité attendue et le périmètre
énergétique. Mo = 10^6 octets ; Mio = 2^20 octets.

## Trois problèmes différents

1. **Poids résidents** : stocker tous les poids utiles en RAM avec cache,
   tokenizer, activations, buffers et runtime.
2. **Poids sur disque** : déplacer une partie des poids hors RAM. Cela ne
   compresse pas le modèle et exige de mesurer les lectures physiques.
3. **Petit modèle outillé** : déplacer les connaissances vers un index ou des
   outils. Cela ne démontre pas une équivalence à un grand généraliste ; la
   qualité se mesure sur une tâche et tous les coûts du système se comptent.

## Arithmétique reproductible, pas preuve d'impossibilité

Pour P paramètres stockés uniformément à b bits : W = P*b/8 octets,
**hors** échelles, métadonnées, tenseurs en haute précision et alignement.

Exemple fictif : dense 7 milliards de paramètres, 4 bits idéaux, contexte non
spécifié (KV exclu de ce calcul), aucun backend, matériel hypothétique,
2026-10-19. W = 3 500 000 000 octets. Avec seulement 100 000 000 octets pour
les poids, le budget moyen serait 0,1143 bit/paramètre. Cela exclut ce stockage
uniforme naïf, pas toute représentation structurée ou tout futur algorithme.

Le ternaire a log2(3) ≈ 1,585 bit/symbole pour le codage compact de toutes les
séquences ternaires. Ce n'est pas automatiquement le format du kernel : un
packing simple utilise 2 bits, et les échelles/autres tenseurs restent à compter.
Des distributions compressibles peuvent se coder autrement, avec un coût de
lecture/décodage à mesurer. « 1,58 bit » ne donne pas directement le RSS.

## Streaming dense : borne conditionnelle de débit

Supposons du décodage autorégressif standard, une séquence, un token à la fois,
sans spéculation, sans réutilisation inter-token des poids non résidents.
Si chaque token exige D octets de **lectures physiques** et que le stockage
soutient au plus B octets/s, alors :

    temps/token >= D/B
    tokens/s <= B/D

Ce n'est ni une mesure ni une borne universelle pour tout algorithme.
Avec le modèle fictif ci-dessus, en accordant généreusement les 100 Mo entiers
aux poids persistants, D >= 3,4 Go/token. À B = 1 Go/s supposé : au moins
3,4 s/token, soit au plus 0,294 token/s. Runtime et KV réduiraient encore la
place persistante. Le calcul, la déquantification et les accès non séquentiels
peuvent ralentir davantage. À puissance moyenne supposée constante de 15 W,
le même scénario coûte au moins 51 J/token ; ce n'est pas une mesure d'énergie.

Le page cache de l'OS peut masquer les lectures et donner l'illusion d'un faible
RSS avec un grand modèle : comparer RSS et mémoire totale attribuée au cgroup.
Le mmap seul ne prouve donc pas un fonctionnement avec 100 Mo de RAM physique.

## Proposition expérimentale S1 — à coordonner avec E0/W5

Pas d'expérience lancée ; pas de nouveau choix architectural.

- Fixer modèle/checksum/licence, quantification, contexte, tokens produits,
  backend/commit, matériel, threads, date et prompt/seed.
- Référence résidente, puis streaming borné sur le même modèle ; séparer
  chargement, prefill, décodage, exécutions froides et chaudes.
- Rapporter RSS maximal, mémoire cgroup (cache fichier inclus), swap,
  octets physiques lus, débit, latence premier token et temps total.
- Interdire le swap pour le test de contrainte physique ; documenter la limite
  cgroup et les éventuels OOM. Ne pas purger globalement les caches d'un hôte partagé.
- Énergie totale et énergie incrémentale séparées ; sans capteur, écrire
  « non mesurée », pas extrapoler un TDP en mesure.
- Qualité sur jeu tenu à l'écart ; streaming exact comparé aux logits/sorties
  de la référence avec tolérance numérique annoncée.
- Critères d'acceptation : plafond mémoire, débit minimal, qualité et budget
  énergétique à obtenir de l'utilisateur avant de déclarer une réussite.

## Questions envoyées aux autres agents

- W5 : qui prend E0 et peut exposer RSS + mémoire cgroup + lectures physiques ?
- W1/W2 : tâche cible et niveau de qualité minimal ?
- W3 : qui audite packing ternaire et buffers de déquantification ?

Ne pas convertir les pourcentages de gains du rapport initial en garanties
universelles. Chaque comparaison doit retrouver sa source, sa configuration et
ses incertitudes. Un généraliste utile sous 100 Mo est **non démontré ici**, pas
mathématiquement impossible sur la seule base de cette arithmétique.
