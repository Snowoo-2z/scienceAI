# Cadrage — KAIROS, régime « ~100 Go de RAM »

**Agent :** KAIROS · **Branche :** `arena/c9070fe4-scienceai` · **Date :** 2026-10-19
**Statut :** travail en cours, chiffres reproductibles, hypothèses signalées.

---

## 1. Ce que j'ai trouvé en arrivant

Le document partagé contient déjà un travail sérieux : 12 onglets, un rapport
v0.1 de 704 lignes (`docs/RECHERCHE-LLM-100MB.md`), un calculateur
(`tools/llm_budget.py`), un protocole multi-agents (`AGENTS.md`) et cinq ADR.
L'agent `9c2ae177` a cadré **100 Mo de RSS**, et l'agent `Silex` a ouvert un
sous-chantier sur le streaming de poids.

Ce travail est cohérent. Mais il répond à une question qui n'est pas la mienne.

## 2. La question des unités, posée franchement

Le brief dit : « faire tourner un **grand LLM** dans **une centaine de Go de
RAM** ». Le document collectif a retenu **100 Mo**. Entre les deux, il y a un
facteur **1 000**. Ce n'est pas un détail de notation : c'est un changement de
régime physique.

| | 100 Mo (document existant) | 100 Go (ce document) |
|---|---|---|
| Paramètres possibles à 4,5 bits | 85 M | 180 B |
| Paramètres possibles à 1,7 bit | 200 M | 420 B |
| Ce qui limite | le runtime (26 Mo incompressibles) | le noyau de calcul et la bande passante |
| Ce qui fait gagner | données, distillation, RAG | sparsité, quantification, kernels |
| Marge d'erreur | ±15 Mo = 15 % du budget | ±2 Go = 2 % du budget |
| Modèle mental | « un assistant spécialisé » | « un gros modèle qui tient sur un bureau » |

**Les deux lectures sont défendables**, mais elles ne produisent ni les mêmes
architectures, ni les mêmes expériences, ni les mêmes budgets. Le document
partagé a déjà tranché pour 100 Mo (ADR-001). Je ne réécris pas ce choix : je
couvre l'autre branche, et je fournis l'équation qui les relie
(`docs/01-LOI-DE-BUDGET.md`).

> **Question pour l'humain, à trancher une fois :** « une centaine » = 100 Mo
> ou 100 Go ? Tant qu'elle n'est pas tranchée, les deux chantiers doivent
> avancer, mais il faut **étiqueter chaque chiffre avec son régime**. Un débit
> de 84 tok/s n'a aucun sens si l'on ne sait pas pour quel budget mémoire.

## 3. Ce que je démontre, et ce que je ne démontre pas

**Démontré par le calcul et la mesure (reproductible dans ce dépôt) :**

1. Un modèle de **372 B de paramètres** dont **6 B seulement sont lus par
   token** tient dans **84,9 Go** avec 32 k de contexte — 2,05 Go lus par
   token, soit 49 tok/s sur un mini-PC à 3 500 € *si* le noyau tient 800 GOP/s.
2. Le même modèle en variante **213 B** tient dans **49,5 Go** (machine 64 Go)
   **à débit identique** : la capacité supplémentaire ne coûte pas un token/s.
3. À 1,58 bit, le goulot **cesse d'être la mémoire** et devient le noyau de
   déquantification. Seuil exact : `GOPS ≥ 16 · BW / bits`. C'est le résultat
   le plus important du document, et il contredit la narration habituelle
   « le décodage est memory-bound ».
4. La dispersion des experts dans plusieurs dizaines de Go **ne coûte rien** :
   mesuré à 105-107 % du débit séquentiel dès 256 Ko de bloc. L'« expert
   clustering » est une fausse piste.
5. En dessous de 4 Ko de bloc, on perd 20 à 65 % de la bande passante. Le vrai
   danger de localité est dans le KV cache et les petits tenseurs, pas dans les
   experts.

**Non démontré, et signalé comme tel :**

- Qu'un modèle de 372 B à 1,58 bit **conserve la qualité** de son enseignant.
  Personne ne l'a publié au-delà de 3 B. C'est l'hypothèse centrale, et elle
  est falsifiable (`docs/05-EXPERIENCES-RISQUES.md`, E-K2).
- Qu'on puisse **entraîner** un tel modèle dans ce projet. Le calcul dit
  ~100 k$ pour 2 T tokens. C'est hors de portée sans financement ; la seule
  voie immédiate est de **quantifier et réparer un modèle existant**.
- Les débits absolus sur le matériel cible : je n'ai mesuré qu'une VM à
  2 vCPU. Les **ratios** sont transférables, les valeurs non.

## 4. Méthode

Cinq règles que je me suis imposées, et que je propose au collectif :

1. **Aucun chiffre sans calcul exécutable.** Tout ce qui est dans ce document
   sort de `tools/hecaton_budget.py`, `tools/bwprobe.c` ou `tools/tgemv.c`.
2. **Séparer mesure, calcul et hypothèse.** Chaque affirmation est étiquetée
   `MESURÉ`, `CALCULÉ` ou `HYPOTHÈSE`. Le document de l'autre branche le fait
   déjà ; je reprends la convention.
3. **Chercher les résultats négatifs.** Le benchmark mémoire a tué une idée
   (l'expert clustering) avant qu'elle ne coûte trois semaines. C'est un
   livrable, pas un échec.
4. **Donner un critère d'échec à chaque idée.** Une idée sans protocole de
   réfutation n'est pas une expérience.
5. **Ne pas promettre « un concurrent de GPT-5 ».** Un 372B-A6B ternaire vise
   la classe gpt-oss-120b / Qwen3-235B, pas la frontière. Le dire tôt évite de
   perdre six mois.

## 5. Périmètre et non-périmètre

**Dans mon périmètre (W1-bis, W3 partiel, W5 partiel) :**

- l'arithmétique du régime 100 Go et la loi de budget ;
- l'architecture HECATON et ses variantes ;
- les noyaux ternaires : mesure, seuils, recommandations ;
- le matériel : ce qui existe, à quel prix, pour quel débit ;
- la faisabilité de l'entraînement, chiffrée ;
- le plan d'expériences et les critères d'abandon.

**Hors périmètre (autres agents) :**

- données, filtrage, distillation (W2) ;
- RAG, outils, grammaire, produit (W4) ;
- arbitrage final Mo/Go et priorités (W6, humain) ;
- le régime 100 Mo, déjà couvert par `9c2ae177` et `Silex`.

## 6. Ce que je demande aux autres agents

| Agent | Question | Pourquoi elle bloque |
|---|---|---|
| `9c2ae177` | la cible est-elle 100 Mo ou 100 Go ? | détermine quel document est le principal |
| `Silex` | le packing ternaire réel (bits effectifs, échelles, buffers) | conditionne le budget mémoire à ±10 % |
| W2 (à ouvrir) | quel corpus pour 1-2 T tokens ? | sans données, 372 B de capacité ne valent rien |
| W4 (à ouvrir) | quel est le vrai coût par réponse utile ? | c'est la métrique qui décide de tout |
| W5 (à ouvrir) | adopter `GOPS ≥ 16·BW/bits` comme test de conception ? | évite d'optimiser la mémoire quand le calcul bloque |

## 7. Où lire quoi

| Fichier | Contenu |
|---|---|
| `docs/01-LOI-DE-BUDGET.md` | les 5 équations, les tableaux, le pont 100 Mo ↔ 100 Go |
| `docs/02-ARCHITECTURE-HECATON.md` | la conception complète, composant par composant |
| `docs/03-NOYAUX-MATERIEL-MESURES.md` | les benchmarks, les seuils, le matériel |
| `docs/04-ENTRAINEMENT-FAISABILITE.md` | trois chemins, trois prix, un verdict |
| `docs/05-EXPERIENCES-RISQUES.md` | E-K1 à E-K10, risques, idées à tester |
| `docs/RESUME.md` | une page, pour le document partagé |
| `results/*.txt` | toutes les sorties brutes |
| `tools/hecaton_budget.py` | le calculateur |
| `tools/bwprobe.c`, `tools/tgemv.c` | les microbenchmarks |
