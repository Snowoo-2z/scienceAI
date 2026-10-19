# Antériorités et hypothèses — mémoire conditionnelle

Silex,19octobre2026. Lecture de sources primaires effectuée pendant S5.
Cette note empêche de rebaptiser un mécanisme connu «nouvelle invention» ou de
transposer un gain GPU/DRAM en promesse CPU/SSD sous200Mo.

## Engram : séparation mémoire/calcul, pas preuve à200Mo

Dépôt officiel `deepseek-ai/Engram`, commit
`fb7f84a21f91223715394a33a1dc24bbfb7f788e`.

- [README](https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/README.md)
- [Code de démonstration](https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/engram_demo_v1.py)
- [Article](https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/Engram_paper.pdf)

Observé dans les sources : IDs dérivés des n-grammes, hachage multi-têtes,
lookup d'embeddings, porte conditionnée par l'état caché, fusion dans le backbone.
Le README précise que le code de démo simule des composants du backbone. Il ne
faut donc pas confondre son exécution avec l'évaluation d'un modèle complet.

Article§2.5 : séparation stockage/calcul ; DRAM hôte, prélecture via PCIe ; NVMe
évoqué dans une hiérarchie de cache. §6.4/Table4 (pages17–18) : mesure de débit
avec une table100B en DRAM hôte, backbones denses4B/8B, GPU H800,512 séquences de
longueur100..1024. Le surcoût rapporté inférieur3% concerne CETTE configuration.
Ce n'est pas une mesure SSD, batch1 CPU, ni200Mo de RAM totale. Les nombres de
qualité des auteurs ne sont pas reproduits par nous.

Ce qui peut guider une nouvelle expérience : adressage connu tôt → prélecture
possible ; mesurer simultanément utilité de la mémoire et coût des accès. Notre
banc S5 donne les IDs à l'avance : même ce routage doit être appris/évalué avant
la conversation. Un accès logique O(1) ne supprime ni latence disque, ni collisions,
ni coût d'entraînement du cœur/gating/tables.

## MoEUT : partage de couches/experts déjà exploré

Dépôt officiel `RobertCsordas/moeut`, commit
`87f04973d3db3ad7cb4ca67a0750241e40210d3c`.

[README](https://github.com/RobertCsordas/moeut/blob/87f04973d3db3ad7cb4ca67a0750241e40210d3c/README.md)

Le README expose des Universal Transformers à experts, `group_size` (couches non
partagées par groupe), nombre d'experts et nombre activé. Implémentation avec
kernel CVMM Triton, exemples CUDA, cache KV optionnel. Nous n'avons ni exécuté ni
validé ces modèles ici. Le partage et le calcul conditionnel ne sont donc pas,
à eux seuls, une nouveauté à revendiquer. Rien dans cette lecture ne valide un
généraliste CPU sous200Mo.

## Où se situe réellement notre question ouverte

Combiner apprentissage de représentations partagées, exceptions indépendantes et
mémoire paginée, avec sélection d'après qualité ET coût total. S5 n'est qu'une
preuve de mécanisme synthétique. Il montre déjà une tension : réserver une base
partagée peut réduire le fichier mais augmenter les défauts de cache. La question
utile n'est pas «combien de paramètres virtuels ?», mais «quelle capacité reste
correcte pour quel coût complet, avec un routage réellement appris ?».

Aucun de nos scripts ne copie le code des dépôts ci-dessus. Le PDF a été consulté
localement via l'API GitHub et pypdf6.1.1 ; pas de PDF redistribué dans ce dépôt.
