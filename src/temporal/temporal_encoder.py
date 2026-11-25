
import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """
    Learnable attention mechanism over temporal features.

    Input:
        (B, T, D)

    Output:
        (B, D)
    """

    def __init__(self, feature_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, x):
        # Calculate attention scores for every timestep
        scores = self.attention(x).squeeze(-1)

        # Normalize scores across time
        weights = torch.softmax(scores, dim=1)

        # Weighted sum of temporal features
        context = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1
        )

        return context, weights


class TemporalEncoder(nn.Module):
    """
    Bidirectional GRU + Temporal Attention.

    Input:
        (B, 32, 512)

    Output:
        Video representation: (B, 512)
        Attention weights:    (B, 32)
    """

    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
        num_layers=1,
        dropout=0.3
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Bidirectional GRU output dimension = 256 * 2
        self.output_dim = hidden_dim * 2

        self.attention = TemporalAttention(
            feature_dim=self.output_dim
        )

        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (B, T, D), got {tuple(x.shape)}"
            )

        if x.shape[-1] != self.gru.input_size:
            raise ValueError(
                f"Expected feature dimension {self.gru.input_size}, "
                f"got {x.shape[-1]}"
            )

        # Model temporal dependencies
        sequence_output, _ = self.gru(x)

        # Normalize temporal representations
        sequence_output = self.norm(sequence_output)

        # Learn the importance of each timestep
        video_features, attention_weights = self.attention(
            sequence_output
        )

        return video_features, attention_weights
