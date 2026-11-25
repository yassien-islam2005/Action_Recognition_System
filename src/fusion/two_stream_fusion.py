
import torch
import torch.nn as nn


class TwoStreamFusion(nn.Module):
    """
    Learnable fusion of Spatial and Temporal video features.

    Inputs:
        spatial_features:  (B, 512)
        temporal_features: (B, 512)

    Output:
        fused_features:    (B, 512)
        fusion_weights:    (B, 2)
    """

    def __init__(self, feature_dim=512, dropout=0.3):
        super().__init__()

        # Learn how much information to take from each stream
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 2)
        )

        self.norm = nn.LayerNorm(feature_dim)

        self.output_dim = feature_dim

    def forward(self, spatial_features, temporal_features):

        if spatial_features.ndim != 2:
            raise ValueError(
                "Spatial features must have shape (B, D)"
            )

        if temporal_features.ndim != 2:
            raise ValueError(
                "Temporal features must have shape (B, D)"
            )

        if spatial_features.shape != temporal_features.shape:
            raise ValueError(
                "Spatial and temporal feature shapes must match"
            )

        # Concatenate both streams to calculate fusion weights
        combined = torch.cat(
            [spatial_features, temporal_features],
            dim=1
        )

        # Calculate two learnable fusion scores
        scores = self.gate(combined)

        # Normalize scores into weights
        weights = torch.softmax(scores, dim=1)

        # Weighted fusion
        fused_features = (
            weights[:, 0:1] * spatial_features
            +
            weights[:, 1:2] * temporal_features
        )

        # Normalize the fused representation
        fused_features = self.norm(fused_features)

        return fused_features, weights
