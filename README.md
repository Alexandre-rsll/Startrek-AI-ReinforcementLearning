# Start Trek

**Projet Epitech - Note obtenue : A**

Start Trek est un projet d'apprentissage par renforcement visant à entraîner un agent capable de faire atterrir un module lunaire de façon autonome sur l'environnement `LunarLander-v3` de Gymnasium. Trois agents à base de réseaux de neurones (DQN, Double DQN, Double DQN + Prioritized Experience Replay) sont implémentés, entraînés et comparés rigoureusement à deux baselines (politique aléatoire et politique heuristique), avec une évaluation statistique multi-seed et une étude d'ablation.

Le projet met l'accent sur la méthodologie expérimentale en RL : protocole d'évaluation reproductible, intervalles de confiance, ablation d'hyperparamètre, et analyse critique des résultats plutôt qu'un simple entraînement unique.

## Sommaire

- [Environnement](#environnement)
- [Agents implémentés](#agents-implémentés)
- [Protocole expérimental](#protocole-expérimental)
- [Résultats](#résultats)
- [Reproduction](#reproduction)
- [Stack technique](#stack-technique)
- [Équipe](#équipe)

## Environnement

L'agent contrôle un module lunaire dans `LunarLander-v3` (contrôle discret) : il observe sa position, sa vitesse, son angle, sa vitesse angulaire et le contact de ses deux jambes avec le sol, et choisit à chaque pas parmi quatre actions (ne rien faire, propulseur gauche, propulseur principal, propulseur droit). L'environnement est considéré comme résolu à partir d'une récompense moyenne supérieure ou égale à 200 sur 100 épisodes d'évaluation.

Deux baselines servent de référence dans `baseline.py` :

- **random** : action tirée uniformément ;
- **heuristic** : contrôleur fait main basé sur l'angle, la vitesse angulaire et la vitesse verticale.

## Agents implémentés

Les trois agents partagent le même réseau de neurones (MLP à deux couches cachées de 64 neurones) et la même boucle d'entraînement générale, mais diffèrent sur le calcul de la cible et la stratégie d'échantillonnage :

- **DQN** : Q-learning profond classique, réseau cible synchronisé périodiquement, mémoire de rejeu uniforme (`deque`).
- **Double DQN** : découple la sélection de l'action (réseau en ligne) de son évaluation (réseau cible) pour réduire le biais de surestimation propre au DQN classique.
- **Double DQN + PER** : ajoute une mémoire de rejeu priorisée (implémentation d'un `SumTree` avec correction par poids d'importance et exposant `alpha`/`beta`) qui rééchantillonne plus souvent les transitions à forte erreur TD.

Chaque agent journalise récompense, longueur d'épisode, décroissance d'epsilon, perte et cause de fin d'épisode (`landed`, `crash`, `sleep`, `out_of_view`), utilisées pour générer automatiquement les graphiques d'entraînement.

## Protocole expérimental

- **Entraînement** : 500 épisodes par agent, avec préremplissage de la mémoire de rejeu avant l'apprentissage.
- **Évaluation finale** : politique gelée (epsilon = 0), moyenne et écart-type sur 100 épisodes (`eval.py`).
- **Multi-seed** (`multi_seed.py`) : chaque agent est ré-entraîné et ré-évalué sur 5 seeds indépendantes, avec calcul d'un intervalle de confiance à 95 % sur la moyenne inter-seeds.
- **Étude d'ablation** : comparaison de trois stratégies de mise à jour du réseau cible de Double DQN (hard update tous les 10 pas, tous les 50 pas, et soft update avec `tau = 0.005`), afin d'isoler l'effet de cet hyperparamètre indépendamment du choix d'algorithme.
- L'ensemble du protocole est exécutable de bout en bout via `repro.sh` et documenté dans `report.pdf`.

## Résultats

| Agent | Récompense moyenne (100 ép.) | Écart-type | Résolu (≥200) |
|---|------------------------------|---|---|
| DQN | 195.10                       | 87.23 | Non, proche du seuil |
| Double DQN | 241.42                       | 52.89 | Oui |
| Double DQN + PER | 270.59                       | 68.01 | Oui |

**Double DQN PER** obtient le meilleur compromis entre performance et stabilité (meilleur score moyen et plus faible variance), et reste le meilleur agent sur l'évaluation multi-seed (237.8 ± 53.6 en moyenne sur 5 seeds). Double DQN + PER atteint le meilleur taux d'atterrissage pendant l'entraînement (67.8 %) mais une évaluation finale plus instable, cohérent avec un possible sur-échantillonnage de transitions rares. L'étude d'ablation confirme que la mise à jour fréquente du réseau cible (tous les 10 pas) est la configuration la plus performante sur ce budget d'entraînement.

Le rapport complet, avec courbes d'entraînement, distribution des causes de fin d'épisode et graphiques multi-seed, est disponible dans [report.pdf](report.pdf).

## Reproduction

```bash
pip install -r requirements.txt

# pipeline complet : baselines, entraînement, évaluation, multi-seed
./repro.sh

# ou individuellement
python baseline.py --config configs/dqn.yaml
python eval.py --config configs/dqn.yaml
python multi_seed.py --config configs/double_dqn_per.yaml
```

## Stack technique

| Composant | Outil |
|---|---|
| Environnement RL | Gymnasium (`LunarLander-v3`), Box2D |
| Deep learning | PyTorch |
| Algorithmes | DQN, Double DQN, Double DQN + PER (SumTree custom) |
| Analyse / visualisation | NumPy, Matplotlib |
| Configuration | YAML |

## Équipe

Projet réalisé en binôme à Epitech :

- Alexandre Rousselle
- Maksymilian Kusy
- manmohit-singh lan
