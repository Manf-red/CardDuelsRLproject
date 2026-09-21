import torch
from gymnasium import spaces
from torch import nn


class MLPCardExtractor(nn.Module):
    """
    Permutation-invariant neural network for processing Bag-of-Cards.
    """
    def __init__(self, card_dim: int, out_dim: int):
        super().__init__()

        self.phi = nn.Sequential(
            nn.Linear(card_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.rho = nn.Sequential(
            nn.Linear(32, out_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape (Batch, Num_Cards, Card_dim)
        card_embeddings = self.phi(x) # Shape: (Batch, Num_cards, 32)

        # Permutation-invariant aggregation
        pooled = torch.sum(card_embeddings, dim=1) # Shape: (Batch, 32)

        out = self.rho(pooled) # Shape: (Batch, out_dim)
        return out

class AttentionInvariantCardExtractor(nn.Module):
    """
    Permutation-invariant network using Self-Attention to model card relations.
    """
    def __init__(self, card_dim: int, out_dim: int, n_heads: int = 4, hidden_dim: int = 64):
        super().__init__()
        
        # Project raw 7D card vectors into a higher dimension for Multi-Head Attention
        self.embedding = nn.Linear(card_dim, hidden_dim)
        
        # Transformer Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=128, 
            batch_first=True, 
            dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (Batch, Num_Cards, Card_dim)
        
        # Project into Embedding
        emb = self.embedding(x) # Shape: (Batch, Num_Cards, hidden_dim)
        
        # Self-Attention computation
        attn_out = self.transformer(emb) # Shape: (Batch, Num_Cards, hidden_dim)
        
        # Permutation-invariant Global Average Pooling
        pooled = torch.mean(attn_out, dim=1) # Shape: (Batch, hidden_dim)
        
        # Final Projection
        out = self.rho(pooled) # Shape: (Batch, out_dim)
        return out

class AttentionFlattenCardExtractor(nn.Module):
    """
    Used for the Hand.
    Uses Self-Attention for synergy, followed by Flattening to preserve Positional Indices.
    """
    def __init__(self, card_dim: int, max_cards: int, out_dim: int, n_heads: int = 4, hidden_dim: int = 64):
        super().__init__()
        
        # Project raw 7D card vectors into a higher dimension for Multi-Head Attention
        self.embedding = nn.Linear(card_dim, hidden_dim)
        
        # Transformer Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=128, 
            batch_first=True, 
            dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Flatten (max_cards * hidden_dim) instead of pooling
        flattened_dim = max_cards * hidden_dim
        self.rho = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (Batch, Num_Cards, Card_dim)
        
        # Project into Embedding
        emb = self.embedding(x) # Shape: (Batch, Num_Cards, hidden_dim)
        
        # Self-Attention computation
        attn_out = self.transformer(emb) # Shape: (Batch, Num_Cards, hidden_dim)
        
        return self.rho(attn_out)

class MultiModalQNetwork(nn.Module):
    """
    Action-value approximator.
    Dynamically builds parallel feature extractors based on the observation space.
    """
    def __init__(self, observation_space: spaces.Dict, n_actions: int, card_extractor_out_dim: int = 64):
        super().__init__()

        self.extractors = nn.ModuleDict()
        total_concat_size = 0

        for key in sorted(observation_space.spaces.keys()):
            subspace = observation_space.spaces[key]

            # Explicitly assert that the shape is not None to satisfy strict type checkers like Pyright.
            assert subspace.shape is not None, f"Observation subspace '{key}' must have a defined shape."

            if len(subspace.shape) == 1:
                # 1D scalars (self_stats, opp_stats)
                input_dim = subspace.shape[0]
                self.extractors[key] = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU()
                )
                total_concat_size += 32

            elif len(subspace.shape) == 2:
                # 2D matrices (Cards)
                max_cards = subspace.shape[0]
                card_dim = subspace.shape[1]
                flattened_dim = max_cards * card_dim
                self.extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(flattened_dim, card_extractor_out_dim),
                    nn.ReLU()
                )
                total_concat_size += card_extractor_out_dim
        
        self.q_value_head = nn.Sequential(
            nn.Linear(total_concat_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Map a batch of dictionary states to a batch of action-value vectors.
        """
        encoded_features = []

        for key in sorted(self.extractors.keys()):
            extractor = self.extractors[key]
            encoded_features.append(extractor(obs_dict[key]))

        fused_vector = torch.cat(encoded_features, dim=1)

        q_values = self.q_value_head(fused_vector)
        return q_values

