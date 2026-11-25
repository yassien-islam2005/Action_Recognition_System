
import torch
import torch.nn as nn

from src.spatial.rgb_backbone import RGBBackbone
from src.temporal.flow_backbone import FlowBackbone
from src.temporal.temporal_encoder import TemporalEncoder
from src.fusion.two_stream_fusion import TwoStreamFusion


class TwoStreamActionRecognition(nn.Module):
    """
    Two-Stream Action Recognition Model for UCF101.

    Spatial input:
        (B, T, 3, H, W)

    Temporal input:
        (B, T, 2, H, W)

    Output:
        (B, num_classes)
    """

    def __init__(
        self,
        num_classes=100,
        num_frames=32,
        frame_batch_size=4,
        hidden_dim=256,
        dropout=0.5,
        pretrained=False
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.frame_batch_size = frame_batch_size

        # Independent spatial and temporal backbones
        self.rgb_backbone = RGBBackbone(
            pretrained=pretrained
        )

        self.flow_backbone = FlowBackbone(
            pretrained=pretrained
        )

        # Independent temporal encoders
        self.rgb_temporal = TemporalEncoder(
            input_dim=512,
            hidden_dim=hidden_dim
        )

        self.flow_temporal = TemporalEncoder(
            input_dim=512,
            hidden_dim=hidden_dim
        )

        # Learnable fusion of both streams
        self.fusion = TwoStreamFusion(
            feature_dim=hidden_dim * 2,
            dropout=0.3
        )

        # Final action classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def _extract_frame_features(self, frames, backbone):
        """
        Extract features from video frames in smaller chunks.

        Input:
            (B, T, C, H, W)

        Output:
            (B, T, 512)
        """

        batch_size, num_frames, channels, height, width = frames.shape

        # Flatten batch and temporal dimensions
        frames = frames.reshape(
            batch_size * num_frames,
            channels,
            height,
            width
        )

        feature_chunks = []

        # Process frames in smaller batches
        for start in range(
            0,
            frames.shape[0],
            self.frame_batch_size
        ):
            end = start + self.frame_batch_size

            frame_chunk = frames[start:end]

            chunk_features = backbone(frame_chunk)

            feature_chunks.append(chunk_features)

        # Combine all extracted features
        features = torch.cat(feature_chunks, dim=0)

        # Restore temporal dimension
        features = features.reshape(
            batch_size,
            num_frames,
            -1
        )

        return features

    def forward(
        self,
        rgb_frames,
        flow_frames,
        return_details=False
    ):

        # Validate input dimensions
        if rgb_frames.ndim != 5:
            raise ValueError(
                "RGB input must have shape (B, T, 3, H, W)"
            )

        if flow_frames.ndim != 5:
            raise ValueError(
                "Flow input must have shape (B, T, 2, H, W)"
            )

        if rgb_frames.shape[:2] != flow_frames.shape[:2]:
            raise ValueError(
                "RGB and Flow batch/time dimensions must match"
            )

        if rgb_frames.shape[1] != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} frames, "
                f"got {rgb_frames.shape[1]}"
            )

        if rgb_frames.shape[2] != 3:
            raise ValueError("RGB input must have 3 channels")

        if flow_frames.shape[2] != 2:
            raise ValueError("Flow input must have 2 channels")

        # 1. Extract spatial frame features
        rgb_features = self._extract_frame_features(
            rgb_frames,
            self.rgb_backbone
        )

        # 2. Extract optical flow frame features
        flow_features = self._extract_frame_features(
            flow_frames,
            self.flow_backbone
        )

        # 3. Model temporal dependencies
        rgb_video_features, rgb_attention = self.rgb_temporal(
            rgb_features
        )

        flow_video_features, flow_attention = self.flow_temporal(
            flow_features
        )

        # 4. Fuse both video representations
        fused_features, fusion_weights = self.fusion(
            rgb_video_features,
            flow_video_features
        )

        # 5. Predict action classes
        logits = self.classifier(fused_features)

        if return_details:
            return {
                "logits": logits,
                "rgb_attention": rgb_attention,
                "flow_attention": flow_attention,
                "fusion_weights": fusion_weights
            }

        return logits

