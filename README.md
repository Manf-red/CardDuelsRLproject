# Card Duels: Deep Q-Network in a High-Variance POMDP

**Abstract:** Developed for Reinforcement Learning exam, this project evaluates a DQN within a custom POMDP modeling a combinatorial card game.

## Mechanics & Environment (`CardDuelsEnv`)

The engine simulates a 1v1 duel where agents manage HP, progressive Mana constraints, and Fatigue damage.

* **Action Masking:** The $2^H$ combinatorial action space uses dynamic masking to eliminate illegal moves.
* **State Encoding:** The observation space isolates player stats and board matrices via Min-Max normalization.

## Agent Architecture & Training

* **MultiModalQNetwork:** Uses Self-Attention and Flattening to calculate card embeddings and relations while preserving spatial indices (where it is deserved).
* **Ablation Studies:** Evaluates the cost of partial observability by comparing POMDP runs against an "Omniscient MDP".
* **Self-Play:** Decouples interactions from optimization, utilizing Shared-Weight Self-Play.
