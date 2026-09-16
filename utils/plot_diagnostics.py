import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scripts.train import TrainingHistory


def plot_training_diagnostics(history: TrainingHistory, save_dir: str = "./plots") -> None:
    """
    Generates a 2x2 grid of publication-ready diagnostic plots to evaluate 
    DQN performance in a combinatorial POMDP environment.
    """
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("DQN Training Diagnostics: Combinatorial POMDP vs. Heuristic", fontsize=18, weight='bold')

    # Helper function for moving averages
    def moving_average(data, window_size=100):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    # --- PLOT 1: Evaluation Returns (Stability & Sample Efficiency) ---
    ax1 = axes[0, 0]
    eval_eps = history.eval_episodes
    mean_rets = np.array(history.eval_mean_returns)
    std_rets = np.array(history.eval_std_returns)
    
    ax1.plot(eval_eps, mean_rets, label="Mean Return", color="#2ca02c", linewidth=2)
    ax1.fill_between(eval_eps, mean_rets - std_rets, mean_rets + std_rets, color="#2ca02c", alpha=0.2, label=r"$\pm 1 \sigma$ (Variance)")
    ax1.set_title("Target Policy: Expected Return & Variance", weight='bold')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Reward")
    ax1.legend(loc="lower right")

    # --- PLOT 2: Evaluation Win Rate (Non-Stationarity & Opponent Modeling) ---
    ax2 = axes[0, 1]
    ax2.plot(eval_eps, history.eval_p0_win_rates, label="DQN Win Rate", color="#1f77b4", linewidth=2.5)
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label="50% Threshold")
    ax2.set_title("Target Policy: Win Rate vs Adaptive Heuristic", weight='bold')
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Win Probability")
    ax2.set_ylim(0.0, 1.0)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.legend(loc="lower right")

    # --- PLOT 3: Network Stability (Combinatorial Action Space) ---
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    
    ax3.plot(history.episodes, history.weight_norms, color="#9467bd", linewidth=2, label=r"Weight $\left\| \theta \right\|_2$")
    ax3_twin.plot(history.episodes, history.grad_norms, color="#ff7f0e", alpha=0.6, linewidth=1.5, label=r"Gradient $\left\| \nabla_\theta L \right\|_2$")
    
    ax3.set_title("Neural Representation: Weights & Gradients", weight='bold')
    ax3.set_xlabel("Episode")
    ax3.set_ylabel(r"Weight $L_2$ Norm", color="#9467bd")
    ax3_twin.set_ylabel(r"Gradient $L_2$ Norm", color="#ff7f0e")
    ax3.grid(False) # Prevent grid overlap
    
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc="upper left")

    # --- PLOT 4: State-Space Horizon (Partial Observability & Fatigue) ---
    ax4 = axes[1, 1]
    raw_lengths = history.train_episode_lengths
    raw_decks = history.train_terminal_deck_sizes
    x_len = np.arange(1, len(raw_lengths) + 1)
    
    # Calculate moving averages (accounting for window shift)
    window = 100
    ma_lengths = moving_average(raw_lengths, window)
    ma_decks = moving_average(raw_decks, window)
    x_ma = x_len[window - 1:]

    ax4.plot(x_ma, ma_lengths, label="Avg Episode Length", color="#d62728", linewidth=2)
    ax4.set_ylabel("Steps per Episode", color="#d62728")
    ax4.tick_params(axis='y', labelcolor="#d62728")
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(x_ma, ma_decks, label="Terminal Deck Size", color="#8c564b", linewidth=2)
    ax4_twin.set_ylabel("Cards Remaining", color="#8c564b")
    ax4_twin.tick_params(axis='y', labelcolor="#8c564b")
    
    ax4.set_title("POMDP Navigation: Game Horizon Dynamics", weight='bold')
    ax4.set_xlabel("Episode")
    ax4.grid(False)

    lines, labels = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save and display
    save_path = os.path.join(save_dir, "dqn_diagnostics_grid.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Diagnostics plot successfully saved to: {save_path}")
    plt.show()


def plot_ablation_study(histories_dict: dict[str, Any], save_path: str = "./plots/ablation_study.png") -> None:
    """
    Generates a comparative 1x2 academic plot overlaying the performance 
    of different state representations.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Ablation Study: Impact of State Representation on Policy Quality", fontsize=16, weight='bold')

    # Color palette to distinctively separate the lines
    colors = sns.color_palette("husl", len(histories_dict))

    ax1, ax2 = axes[0], axes[1]

    for idx, (label, history) in enumerate(histories_dict.items()):
        color = colors[idx]
        eval_eps = history.eval_episodes
        
        # --- PLOT 1: Expected Return (Policy Quality & Stability) ---
        mean_rets = np.array(history.eval_mean_returns)
        #std_rets = np.array(history.eval_std_returns)
        
        # Calculate moving average for smoother visual trends
        window = max(1, len(mean_rets) // 10)
        ma_rets = np.convolve(mean_rets, np.ones(window)/window, mode='valid')
        ma_eps = eval_eps[window-1:]
        
        ax1.plot(ma_eps, ma_rets, label=label, color=color, linewidth=2.5)
        # Optional: Add variance shading for the raw data
        # ax1.fill_between(eval_eps, mean_rets - std_rets, mean_rets + std_rets, color=color, alpha=0.1)

        # --- PLOT 2: Absolute Win Rate vs Heuristic ---
        win_rates = np.array(history.eval_p0_win_rates)
        ma_wins = np.convolve(win_rates, np.ones(window)/window, mode='valid')
        ax2.plot(ma_eps, ma_wins, label=label, color=color, linewidth=2.5)

    # Format Expected Return Plot
    ax1.set_title(r"Target Policy: Moving Average Expected Return", weight='bold')
    ax1.set_xlabel("Training Episode")
    ax1.set_ylabel("Cumulative Reward")
    ax1.legend(loc="lower right")

    # Format Win Rate Plot
    ax2.set_title(r"Target Policy: Win Probability vs Adaptive Heuristic", weight='bold')
    ax2.set_xlabel("Training Episode")
    ax2.set_ylabel("Win Probability")
    ax2.set_ylim(0.0, 1.0)
    ax2.axhline(0.5, color='black', linestyle='--', alpha=0.5, label="50% Threshold")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.legend(loc="lower right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Ablation study plot saved to: {save_path}")
    plt.show()

def plot_hyperparameter_variance(histories_dict: dict[str, Any], save_path: str = "./plots/variance_analysis.png") -> None:
    """
    Generates a 1x2 academic plot specifically designed to visualize 
    training stability and statistical variance across hyperparameter configurations.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Hyperparameter Ablation: Buffer Size & Target Sync Impact on POMDP Variance", fontsize=16, weight='bold')

    # Use a high-contrast palette for clear distinction
    colors = sns.color_palette("Set1", len(histories_dict))

    ax1, ax2 = axes[0], axes[1]

    for idx, (label, history) in enumerate(histories_dict.items()):
        color = colors[idx]
        eval_eps = np.array(history.eval_episodes)
        
        # Extract metrics
        mean_rets = np.array(history.eval_mean_returns)
        std_rets = np.array(history.eval_std_returns)
        win_rates = np.array(history.eval_p0_win_rates)
        
        # Calculate moving averages (Smoothing window = 10% of the data points)
        window = max(1, len(mean_rets) // 10)
        ma_rets = np.convolve(mean_rets, np.ones(window)/window, mode='valid')
        ma_std = np.convolve(std_rets, np.ones(window)/window, mode='valid') 
        ma_wins = np.convolve(win_rates, np.ones(window)/window, mode='valid')
        ma_eps = eval_eps[window-1:]
        
        # --- PLOT 1: Expected Return with Explicit Standard Deviation Bands ---
        ax1.plot(ma_eps, ma_rets, label=label, color=color, linewidth=2.5)
        
        # Explicitly shade the variance (std_rets) to satisfy the analysis requirement.
        # Alpha is set to 0.15 so overlapping regions visually combine rather than block each other.
        ax1.fill_between(ma_eps, ma_rets - ma_std, ma_rets + ma_std, color=color, alpha=0.15, edgecolor="none")

        # --- PLOT 2: Win Rate with Background Noise Visualization ---
        # Plot raw, noisy data in the background to show the exact amplitude of the oscillations
        ax2.plot(eval_eps, win_rates, color=color, alpha=0.15, linewidth=1.0)
        
        # Plot the stable moving average in the foreground
        ax2.plot(ma_eps, ma_wins, label=label, color=color, linewidth=2.5)

    # Format Expected Return Plot
    ax1.set_title(r"Expected Return with $\pm 1\sigma$ Variance Bands", weight='bold')
    ax1.set_xlabel("Training Episode")
    ax1.set_ylabel("Cumulative Reward")
    ax1.legend(loc="lower right")

    # Format Win Rate Plot
    ax2.set_title(r"Win Probability (Raw Oscillations vs. Moving Average)", weight='bold')
    ax2.set_xlabel("Training Episode")
    ax2.set_ylabel("Win Probability vs Heuristic")
    ax2.set_ylim(0.0, 1.0)
    ax2.axhline(0.5, color='black', linestyle='--', alpha=0.5, label="50% Threshold")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.legend(loc="lower right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Variance analysis plot saved successfully to: {save_path}")
    plt.show()