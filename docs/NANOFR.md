# NanoFR — un modèle de langue entraîné et servi ici, sans GPU

> Résultat concret du 2026-10-19 : la chaîne complète **corpus → tokenizer →
> entraînement numpy → GGUF → quantification → llama.cpp → navigateur** tourne
> sur 2 vCPU et 4 Go de RAM. Le modèle est petit (6,5 M de paramètres) ; ce qui
> est démontré, c'est le **chemin de production**, pas la qualité d'un généraliste.

---

## 1. Pourquoi ce modèle existe

Les documents du projet décrivaient une cible (270 M de paramètres dans 200 Mo)
sans jamais montrer un seul token sorti d'un modèle. Une cible non exécutée est
une hypothèse. NanoFR est l'expérience E0bis : prouver que la chaîne tient, et
mesurer chaque étape avec des chiffres reproductibles.

Trois contraintes ont guidé les choix :

1. **Aucun GPU, 2 vCPU, 4 Go de RAM.** L'entraînement doit tenir sous 1 Go.
2. **Aucune dépendance lourde.** numpy + sentencepiece seulement ; pas de
   PyTorch, pas de CUDA.
3. **Le modèle doit sortir en GGUF** et être exécutable par llama.cpp, sinon la
   démonstration ne prouve rien du chemin de déploiement.

## 2. Corpus

| Source | Licence | Volume utilisé |
|---|---|---|
| isosceles (Maupassant FR + Poe traduit par Baudelaire) | CC-BY 4.0 | 381 textes |
| ELTeC-fra, 100 romans 1840–1920 | CC-BY 4.0 | 22 romans (budget 9 M caractères) |

Après nettoyage (NFC, espaces, lignes vides) et mélange : **16,02 M de
caractères** d'entraînement, 0,21 M de validation, 403 documents.

## 3. Tokenizer

SentencePiece BPE entraîné sur le corpus : **8 192 pièces**, `byte_fallback=True`
(tout texte reste encodable), BOS=1, EOS=2. Rendement mesuré : **3,80
caractères par token**.

Le choix de 8 192 pièces plutôt que 32 000 ou 262 144 est un choix de calcul :
la projection de sortie coûte `2 × d × vocab` par token, soit 4,2 MFLOP avec
8 192 pièces contre 16,4 avec 32 000. Sur un petit modèle, **le vocabulaire
coûte plus cher que les couches** — c'est exactement le mécanisme qui empêche
Gemma 3 270M de tenir dans 200 Mo en quantification standard.

## 4. Architecture

| Paramètre | Valeur |
|---|---:|
| Paramètres | 6,52 M |
| Couches | 6 |
| d_model | 256 |
| Têtes Q / KV | 4 / 2 (GQA) |
| head_dim | 64 |
| FFN (SwiGLU) | 704 |
| Contexte | 256 |
| Embeddings | partagés (weight tying) |
| Norme | RMSNorm (eps 1e-5) |
| Position | RoPE split-half, base 10 000 |

L'implémentation (`tools/nanofr_train.py`) est un transformer « architecture
LLaMA » écrit à la main en numpy : forward et backward dérivés à la main, AdamW,
clip global à 1,0, warmup 60 pas puis cosinus. **Aucun autograd.**

Les gradients ont été vérifiés par différences finies (erreur < 5 % sur
`wq`, `w_down`, `tok_embd`) ; cette vérification a révélé un bug réel dans la
rétropropagation de RMSNorm (utilisation de la sortie normalisée au lieu de
l'entrée), qui faisait exploser les gradients d'un facteur ~10⁴.

## 5. Entraînement (mesuré)

| Grandeur | Valeur |
|---|---|
| Pas d'optimisation | 2 400 |
| Batch × contexte | 8 × 256 = 2 048 tokens/pas |
| Tokens vus | 4,92 M (≈ 1,17 époque) |
| Débit | 1,4–2,0 k tokens/s (2 vCPU, AVX-512) |
| Durée | **59,7 min** |
| Pic RSS | **824 Mo** (contrainte : 1 Go — la marge est de 20 %) |
| Loss (train) | 9,06 → **4,19** |

## 6. Résultats mesurés

### Qualité

| Mesure | Valeur |
|---|---:|
| Loss validation | 4,404 nats/token |
| Perplexité validation | **81,8** |
| Bits par caractère | **1,670** |
| Loss entraînement | 4,353 (écart faible : peu de surapprentissage) |

Autrement dit : le modèle prédit le prochain token français avec 1,67 bit par
caractère, contre 4,75 bits pour du hasard uniforme sur les 256 octets et ~1,1 à
1,4 bit pour un bon modèle de caractères entraîné sur beaucoup plus de texte.

### Générations brutes (graine fixée, température 0,8, top-k 40)

| Début | Continuation (sans retouche) |
|---|---|
| Il était une fois, au bord de la Seine, | il s'enait, un grand homme à une heure, dans la route de ce qui s'écria : « Ma fille ! » Et elle s'approcha de la nuit de l'autre, de sa robe d'abord, et un petit peu, une voix plus fort douce ne futcure-là ! |
| Le vieux pêcheur regarda la mer et | se mit à voix. Le matin, d'un de ces dames qui se mêlée de l'autre. Elle avait l'une heure, une grande porte à l'autre et le père ne pouvait avoir été sans cesse à l'autre, un autre côté du matin un peu d'heure. |
| Elle entra dans la salle, et | il lui dit : « Viens-nous donc ! » Le vieillard fit un peu tout basuroë et il me dit : « Et je t'aimerz-elle en se mit à peine. Les femmes-unée, de l'autre, une petite fille, qui n'estéculergmattic |
| La nuit tombait sur le village ; | et la nuit était la tête d'un de la cheminée, d'une chose. C'était, en effet, un grand-là, une espèce de ses mains, un instant, un homme, et son petit père avait d'ailleurs, un bruit, qui se leva l'autre, l'autre, le cœur, les yeux |
| — Mon ami, dit-il, | je suis un jeune homme. C'est beau moment de vous n'était ! Nous me l'avez-vous l'approche de la vérité, sans doute. Elle était là, en effet, il ne peut-il faire des mains, elle a deux. |
| Ce matin-là, le soleil | était encore plus simple. Il n'est qu'une fille ! — C'était bien qu'un des gens, et les gens de la femme que je ne saurait pas une heure d'amuleur dans laquelle il le jeune homme, n'il a eu là-là, je suis l'exquisañant |

**Lecture honnête.** La langue est reconnaissable : orthographe, ponctuation,
dialogue, subordonnées, lexique de Maupassant. Mais le modèle **invente des
mots** (« futcure », « basuroë », « n'estéculergmattic ») et perd le fil
sémantique au bout de deux phrases. C'est le comportement attendu d'un modèle de
6,5 M de paramètres ayant vu 5 M de tokens : il a appris la *forme* du français,
pas son contenu. Le présenter comme un assistant serait malhonnête.

### Tailles et empreinte mémoire

| Fichier | Taille | Pic RSS mesuré (llama-cli) |
|---|---:|---:|
| F32 | 26,3 Mo | 47,2 Mo |
| Q8_0 | **6,9 Mo** | **28,9 Mo** |
| Q4_0 | 4,3 Mo | 28,5 Mo |

Le plancher de 28,5 Mo est le **runtime llama.cpp** (contexte 256, KV minuscule),
pas le modèle : c'est cohérent avec les 26 Mo de surcoût estimés dans
`ARCHITECTURE-200MB-WEB.md` §4.

### Débit

| Chemin | Débit |
|---|---:|
| llama-server, 2 cœurs, Q8 | **1 358 à 1 922 tok/s** |
| Via l'application web | ~1 075 tok/s |
| Application seule (page + relais) | 20,2 Mo de RSS |

Ces débits sont ceux d'un modèle de 6,5 M de paramètres : ils ne se transposent
pas à 270 M, où le décodage est limité par la bande passante mémoire (voir la
loi `GOPS ≥ 16·BW/bits` citée par KAIROS).

## 7. Déploiement

```
llama-server -m data/nanofr_q8_0.gguf -c 256 -t 2 --host 127.0.0.1 --port 8081
python3 app/server.py --port 8000 --backend http://127.0.0.1:8081
```

L'application (`app/`) est un serveur HTTP de la bibliothèque standard : une page,
un endpoint `/api/complete` qui relaie vers llama-server. Aucune dépendance.

## 8. Ce que ce résultat prouve — et ce qu'il ne prouve pas

**Prouvé :** la chaîne complète fonctionne, chaque étape est mesurée, les
fichiers sont reproductibles, le coût en RAM est connu (28,9 Mo de RSS pour le
modèle Q8, application 20,2 Mo), et l'entraînement tient réellement sous 1 Go
(824 Mo de pic).

**Non prouvé :** qu'un modèle de 6,5 M de paramètres soit utile. Il imite un
style, il ne répond pas à des questions. La cible du projet reste un généraliste
pré-entraîné de 270 M quantifié en 4 bits (voir `ARCHITECTURE-200MB-WEB.md`),
que ce banc ne remplace pas : il en valide l'outillage.
