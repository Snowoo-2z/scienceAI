# NanoFR — le modèle tourne, voici ce qu'il écrit

**Agent-Recherche · 2026-10-19 · résultat concret, mesuré, reproductible**

## En une phrase

J'ai entraîné un modèle de langue français de 6,5 millions de paramètres, de zéro,
sur ce mini-PC (2 cœurs, 4 Go), sans GPU, sous 1 Go de RAM — puis je l'ai exporté
en GGUF, quantifié, et servi dans une page web qui tourne en ce moment.

## Ce qui tourne maintenant

- `llama-server` exécute le modèle quantifié (Q8, **6,9 Mo** sur disque) sur le port 8081.
- Une petite application (20,2 Mo de RSS) sert la page et relaie les requêtes.
- L'aperçu du port 8000 est ouvert : tapez un début de phrase, le modèle continue.

## Les chiffres, tous mesurés

| Étape | Résultat |
|---|---|
| Corpus | 403 textes français du domaine public (CC-BY) : Maupassant, Poe traduit par Baudelaire, 100 romans ELTeC 1840–1920 |
| Tokenizer | BPE entraîné sur le corpus, 8 192 pièces, 3,80 caractères par token |
| Modèle | 6,52 M de paramètres, 6 couches, architecture LLaMA, écrit à la main en numpy |
| Entraînement | 2 400 pas, **59,7 min**, pic de mémoire **824 Mo** (limite : 1 Go) |
| Qualité | perplexité de validation **81,8**, soit **1,67 bit par caractère** |
| Quantification | F32 26,3 Mo → Q8 **6,9 Mo** → Q4 4,3 Mo |
| Inférence | **28,9 Mo de RSS**, **1 358 à 1 922 tokens/s** sur deux cœurs |
| Application | 20,2 Mo de RSS, latence de génération ~1 s pour 60 tokens |

## Ce qu'il écrit (brut, sans retouche)

> **Il était une fois, au bord de la Seine,** il s'enait, un grand homme à une heure,
> dans la route de ce qui s'écria : « Ma fille ! » Et elle s'approcha de la nuit de
> l'autre, de sa robe d'abord, et un petit peu, une voix plus fort douce ne futcure-là !

> **La nuit tombait sur le village ;** et la nuit était la tête d'un de la cheminée,
> d'une chose. C'était, en effet, un grand-là, une espèce de ses mains, un instant,
> un homme, et son petit père avait d'ailleurs, un bruit, qui se leva l'autre…

> **— Mon ami, dit-il,** je suis un jeune homme. C'est beau moment de vous n'était !
> Nous me l'avez-vous l'approche de la vérité, sans doute.

## Ce qu'il faut en penser, sans se raconter d'histoires

**Ce qui marche :** la chaîne complète — corpus, tokenizer, entraînement, export
GGUF, quantification, serveur, navigateur — fonctionne, et chaque étape est
mesurée. Le modèle a appris la forme du français : orthographe, ponctuation,
dialogue, subordonnées, lexique de Maupassant.

**Ce qui ne marche pas encore :** il invente des mots (« futcure », « basuroë »)
et perd le fil au bout de deux phrases. C'est normal à cette taille : 6,5 M de
paramètres et 5 M de tokens vus, c'est mille fois moins qu'un modèle utile.
**Ce n'est pas un assistant, et je ne le présenterai pas comme tel.**

**Ce que ça prouve pour le projet :** le chemin de production est validé de bout
en bout, et le coût mémoire d'un modèle de cette famille est connu (runtime
llama.cpp : 28,5 Mo incompressibles). La cible du projet — un généraliste de
270 M de paramètres dans 200 Mo — n'est pas atteinte par ce modèle : elle dépend
d'un modèle pré-entraîné que cet environnement ne peut pas télécharger.

## Ce que j'ai changé dans mes conclusions après avoir lu les autres

1. **MNÉMOSYNE (E-M0b)** : lire un fichier gonfle `memory.current` de la taille
   du fichier, même quand `VmRSS` reste bas. Conséquence : le plafond de 200 Mo
   doit porter sur le **RSS du processus**, pas sur le cgroup — sinon le noyau
   évince les pages du modèle et le débit s'effondre (−25 % mesurés).
2. **Silex** : la contrainte est bien la RAM (100–200 Mo), le disque est permis.
   Donc un GGUF sur disque + un cache borné est une stratégie légitime.
3. **KAIROS** : la loi `GOPS ≥ 16 × bande passante / bits par poids` explique
   pourquoi mes 1 900 tok/s ne se transposent pas à 270 M : le décodage devient
   limité par la mémoire, pas par le calcul.

## Trois choses que je ne peux pas faire ici

1. **Télécharger Gemma 3 270M** : l'environnement n'autorise que GitHub, npm et
   PyPI. Les 183–193 Mo de RSS annoncés pour la cible restent donc **calculés,
   pas mesurés**.
2. **Écrire dans le Google Doc « Project LLM »** : le connecteur Google Docs a
   été retiré. Je lis le Doc (export Drive) et j'y ai déposé un commentaire, mais
   je ne peux plus créer d'onglet. Si vous le remettez, je reprends la
   synchronisation normalement.
3. **Faire mieux que ce modèle sans y passer des heures** : la prochaine marche
   utile n'est pas « entraîner plus longtemps NanoFR », c'est « exécuter un
   modèle pré-entraîné de 135–270 M et le quantifier ».

## Ce que je vous demande

Une seule décision, et elle débloque la suite : **voulez-vous que je prépare la
pile de déploiement complète (Docker + llama-server + base BM25 + page de chat)
pour que vous puissiez la lancer chez vous avec un vrai modèle téléchargé sur
votre PC ?** C'est la seule façon d'obtenir les 183–193 Mo mesurés et une qualité
réellement utilisable, puisque cet environnement n'a pas accès aux modèles.
