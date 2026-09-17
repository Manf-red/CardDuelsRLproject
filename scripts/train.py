import logging
import os
from dataclasses import dataclass, field
from glob import glob

import numpy as np

logger = logging.getLogger(__name__)

from agents.dqn_agent import DQNAgent
from agents.heuristic_agent import HeuristicOpponent
from env.environment import CardDuelsEnv


@dataclass
class TrainingHistory:
    """Encapsulates diagnostic metadata for plotting and formal evaluation."""
    episodes: list[int] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    weight_norms: list[float] = field(default_factory=list)
    
    # Granular Training Diagnostics
    train_episode_lengths: list[int] = field(default_factory=list)
    train_terminal_deck_sizes: list[int] = field(default_factory=list)
    
    # Formal Evaluation Metrics (Target Policy)
    eval_episodes: list[int] = field(default_factory=list)
    eval_p0_win_rates: list[float] = field(default_factory=list)
    eval_mean_returns: list[float] = field(default_factory=list)
    eval_std_returns: list[float] = field(default_factory=list)

def evaluate_agent(
    agent: DQNAgent, 
    env: CardDuelsEnv, 
    heuristic_agent: HeuristicOpponent, 
    n_episodes: int = 100, 
    max_steps: int = 200, 
    seed: int | None = None
) -> tuple[float, float, float]:
    """Runs a batch of greedy evaluation episodes (epsilon = 0.0)."""
    returns = np.zeros(n_episodes, dtype=np.float64)
    wins = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed if ep == 0 else None)
        ep_return = 0.0
        
        for step in range(max_steps):
            if env.board is None: break
            
            current_player = env.board.active_p_idx
            action_mask = info["action_mask"]

            if current_player == 0:
                action = agent.select_action(obs, action_mask, is_greedy=True)
            else:
                action = heuristic_agent.select_action(obs, action_mask)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            obs, info = next_obs, next_info

            # Accumulate reward only for the DQN
            if current_player == 0:
                ep_return += reward

            if terminated or truncated:
                # In env.step, reward is 1.0 if the CURRENT player wins.
                if (reward == 1.0 and current_player == 0) or (reward == -1.0 and current_player == 1):
                    wins += 1
                break
                
        returns[ep] = ep_return

    mean_return = float(returns.mean())
    std_return = float(returns.std())
    win_rate = wins / n_episodes
    
    return mean_return, std_return, win_rate


def train_against_heuristic(agent: DQNAgent, env: CardDuelsEnv, heuristic_agent: HeuristicOpponent, episodes: int, *, max_steps: int = 200, train_freq: int = 4, log_every: int = 100, seed: int | None = 0, save_dir: str = "./checkpoints/heuristic/") -> TrainingHistory:
    """
    Executes the main Reinforcement Learning loop.
    """
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    os.makedirs(save_dir, exist_ok=True)
    
    history = TrainingHistory()
    win_counts = {-1: 0, 0: 0, 1: 0} # -1: Draw, 0: Player 0, 1: Player 1

    global_step = 0
    
    for episode in range(1, episodes + 1):
        obs, info = env.reset(seed=seed if episode == 1 else None)
        pending_transitions: dict[int, tuple[dict[str, np.ndarray], int]] = {}
        
        for step in range(max_steps):
            global_step += 1

            if env.board is None:
                break
                
            current_player = env.board.active_p_idx
            action_mask = info["action_mask"]

            if current_player == 0:
                action = agent.select_action(obs, action_mask)
            else:
                action = heuristic_agent.select_action(obs, action_mask)

            # Store the current state and action for the next self turn. We cannot store the reward or next_state until this player's NEXT turn (or until the game ends).
            pending_transitions[current_player] = (obs, action)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            next_mask = next_info["action_mask"]

            if step == max_steps - 1:
                truncated = True

            if terminated or truncated:
                bellman_done = bool(terminated)
                if env.board is not None:
                    history.train_episode_lengths.append(step + 1)
                    history.train_terminal_deck_sizes.append(len(env.board.players[0].deck.cards)) # Deck size of Player 0 (the DQN agent)
                # If Player 0 ends the game
                if current_player == 0:
                    s, a = pending_transitions[0]
                    agent.store_transition(s, a, reward, next_obs, bellman_done, next_mask)
                # If Player 1 ends the game, Player 0 receives the inverse reward
                elif 0 in pending_transitions:
                    s, a = pending_transitions[0]
                    agent.store_transition(s, a, -reward, next_obs, bellman_done, next_mask)
                    
                win_counts[current_player if reward == 1.0 else (1 - current_player) if reward == -1.0 else -1] += 1
            else:
                next_player = env.board.active_p_idx
                # Only attribute non-terminal rewards when control returns to the DQN (Player 0)
                if next_player == 0 and 0 in pending_transitions:
                    s_prev, a_prev = pending_transitions[0]
                    agent.store_transition(s_prev, a_prev, 0.0, next_obs, False, next_mask)

            # Only optimize the DQN, the heuristic is static
            if global_step % train_freq == 0:
                agent.optimize_model()

            obs, info = next_obs, next_info

            if terminated or truncated:
                break

        agent.end_episode()
        heuristic_agent.end_episode(episode / episodes)  # Gradually reduce randomness in the heuristic opponent
                    
        if episode % log_every == 0:
            # Evaluate the deterministic target policy
            cached_temp = heuristic_agent.temperature
            heuristic_agent.temperature = 0.01
            mean_ret, std_ret, eval_win_rate = evaluate_agent(agent, env, heuristic_agent, n_episodes=20, seed=seed)
            heuristic_agent.temperature = cached_temp
            
            # Store Evaluation Metrics
            history.eval_episodes.append(episode)
            history.eval_p0_win_rates.append(eval_win_rate)
            history.eval_mean_returns.append(mean_ret)
            history.eval_std_returns.append(std_ret)
            
            # Store Training Metrics
            grad_norm = agent.grad_norms[-1] if agent.grad_norms else 0.0
            weight_norm = agent.weight_norms[-1] if agent.weight_norms else 0.0
            history.episodes.append(episode)
            history.epsilons.append(agent.epsilon)
            history.grad_norms.append(grad_norm)
            history.weight_norms.append(weight_norm)
            
            # Print Diagnostics
            avg_len = np.mean(history.train_episode_lengths[-log_every:])
            avg_deck = np.mean(history.train_terminal_deck_sizes[-log_every:])

            snapshot_path = os.path.join(save_dir, f"dqn_heuristic_{episode:05d}.pth")
            agent.save_weights(snapshot_path)
            
            logger.info(
                f"Ep {episode:05d} | Eval Win: {eval_win_rate:.2f} | Eval Ret: {mean_ret:+.2f} ± {std_ret:.2f} | "
                f"Eps: {agent.epsilon:.3f} | |w|: {weight_norm:.2f} | Avg Len: {avg_len:.1f} | Avg Deck: {avg_deck:.1f} | Saved: {os.path.basename(snapshot_path)}"
            )

    return history


def train_self_play(agent: DQNAgent, env: CardDuelsEnv, evaluation_benchmark_agent: HeuristicOpponent, episodes: int, *, max_steps: int = 200, train_freq: int = 4, log_every: int = 100, seed: int | None = 0, save_dir: str = "./checkpoints/selfplay/") -> TrainingHistory:
    """
    Executes the main Reinforcement Learning loop using Shared-Weight Self-Play.
    """
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    os.makedirs(save_dir, exist_ok=True)
    
    # Tracking metrics
    history = TrainingHistory()
    win_counts = {-1: 0, 0: 0, 1: 0} # -1: Draw, 0: Player 0, 1: Player 1

    global_step = 0
    
    for episode in range(1, episodes + 1):
        obs, info = env.reset(seed=seed if episode == 1 else None)
        # Temporal buffer to correctly align the MDP transitions for alternating turns: {player_index: (state, action)}
        pending_transitions: dict[int, tuple[dict[str, np.ndarray], int]] = {}
        
        for step in range(max_steps):
            global_step += 1

            if env.board is None:
                break
                
            current_player = env.board.active_p_idx
            action_mask = info["action_mask"]
            
            # Action Selection
            action = agent.select_action(obs, action_mask, is_greedy=False)  # Shared-Weight Self-Play: Both players use the same DQN weights
            
            # Store the current state and action for the next self turn. We cannot store the reward or next_state until this player's NEXT turn (or until the game ends).
            pending_transitions[current_player] = (obs, action)
            
            # Environment Step (play_turn for current player, start_turn for next player)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            next_mask = next_info["action_mask"]
            if step == max_steps - 1:
                truncated = True

            # Replay Buffer Storage & Reward Attribution
            if terminated or truncated:
                bellman_done = bool(terminated)
                if env.board is not None:
                    history.train_episode_lengths.append(step + 1)
                    history.train_terminal_deck_sizes.append(len(env.board.players[0].deck.cards))

                
                s, a = pending_transitions[current_player]
                agent.store_transition(s, a, reward, next_obs, bellman_done, next_mask)
                
                # The opposing player gets the inverse reward
                opponent = 1 - current_player
                if opponent in pending_transitions:
                    s_opp, a_opp = pending_transitions[opponent]
                    agent.store_transition(s_opp, a_opp, -reward, next_obs, bellman_done, next_mask)
                
                # Record winner for logging
                if reward == 1.0:
                    win_counts[current_player] += 1
                elif reward == -1.0:
                    win_counts[opponent] += 1
                else:
                    win_counts[-1] += 1
            else:
                # If the game continues, the current active player is now the OPPONENT.
                # If the opponent has a pending transition from their previous turn, we now know they survived the current player's attack. 
                # Reward is 0.0 (non-terminal), and the new state is next_obs.
                next_player = env.board.active_p_idx # The board.active player has changed at the end of env.step
                if next_player in pending_transitions:
                    s_prev, a_prev = pending_transitions[next_player]
                    agent.store_transition(s_prev, a_prev, 0.0, next_obs, False, next_mask)

            # Optimize the model
            if global_step % train_freq == 0:
                agent.optimize_model()

            obs, info = next_obs, next_info

            if terminated or truncated:
                break
        
        agent.end_episode()

        if episode % log_every == 0:
            # Evaluate against the STATIC Heuristic, not itself
            cached_temp = evaluation_benchmark_agent.temperature
            evaluation_benchmark_agent.temperature = 0.01
            mean_ret, std_ret, eval_win_rate = evaluate_agent(agent, env, evaluation_benchmark_agent, n_episodes=20, seed=seed)
            evaluation_benchmark_agent.temperature = cached_temp
            
            history.eval_episodes.append(episode)
            history.eval_p0_win_rates.append(eval_win_rate)
            history.eval_mean_returns.append(mean_ret)
            history.eval_std_returns.append(std_ret)
            
            grad_norm = agent.grad_norms[-1] if agent.grad_norms else 0.0
            weight_norm = agent.weight_norms[-1] if agent.weight_norms else 0.0
            history.episodes.append(episode)
            history.epsilons.append(agent.epsilon)
            history.grad_norms.append(grad_norm)
            history.weight_norms.append(weight_norm)
            
            avg_len = np.mean(history.train_episode_lengths[-log_every:])
            avg_deck = np.mean(history.train_terminal_deck_sizes[-log_every:])
            
            snapshot_path = os.path.join(save_dir, f"dqn_selfplay_{episode:05d}.pth")
            agent.save_weights(snapshot_path)
            
            logger.info(
                f"Self-Play Ep {episode:05d} | Static Eval Win: {eval_win_rate:.2f} | Eval Ret: {mean_ret:+.2f} | "
                f"Eps: {agent.epsilon:.3f} | |w|: {weight_norm:.2f} | Avg Len: {avg_len:.1f} | Avg Deck: {avg_deck:.1f} | Saved: {os.path.basename(snapshot_path)}"
            )

    return history


def load_latest_checkpoint(agent: DQNAgent, save_dir: str = "./checkpoints") -> str:
    """
    Finds and loads the highest-episode PyTorch checkpoint in the given directory.
    """
    search_pattern = os.path.join(save_dir, "dqn_heuristic_*.pth")
    checkpoints = glob(search_pattern)
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {save_dir}")
    
    # Select the highest episode checkpoint.
    latest_checkpoint = max(checkpoints)
    
    agent.load_weights(latest_checkpoint)
    logger.info(f"Successfully loaded advanced weights from: {latest_checkpoint}")
    return latest_checkpoint