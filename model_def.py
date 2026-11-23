
"""Standalone inference model for the three-branch deepfake detector.

Usage:
    from model_def import load_detector, predict
    model, config = load_detector("path/to/bundle", device="cpu")
    probs, gates = predict(model, face, eyes, dct)   # tensors, see model_config.json
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


class FaceBranch(nn.Module):
    def __init__(self, embed_dim=256, dropout=0.4, pretrained=False):
        super().__init__()
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1536, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.features(x)
        x = self.pool(x)
        return self.projection(x)


class EyeBranch(nn.Module):
    def __init__(self, embed_dim=256, dropout=0.4, pretrained=False):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        backbone = models.resnet34(weights=weights)

        self.features = nn.Sequential(*list(backbone.children())[:-1])

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.features(x)
        return self.projection(x)


class DCTBranch(nn.Module):
    def __init__(self, embed_dim=256, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.features(x)
        return self.projection(x)


class AdaptiveGatedFusion(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 3 // 2),
            nn.BatchNorm1d(embed_dim * 3 // 2),
            nn.GELU(),
            nn.Linear(embed_dim * 3 // 2, 3),
        )

    def forward(self, face, eyes, dct):
        concat_features = torch.cat([face, eyes, dct], dim=1)
        scores = self.gate(concat_features)
        weights = torch.softmax(scores, dim=1)
        fused = (face * weights[:, 0:1]) + (eyes * weights[:, 1:2]) + (dct * weights[:, 2:3])
        return fused, weights


class DeepfakeDetector(nn.Module):
    def __init__(self, embed_dim=256, num_classes=1, dropout=0.4, pretrained=False):
        super().__init__()
        self.face_branch = FaceBranch(embed_dim, dropout, pretrained)
        self.eye_branch = EyeBranch(embed_dim, dropout, pretrained)
        self.dct_branch = DCTBranch(embed_dim, dropout)

        self.fusion = AdaptiveGatedFusion(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, face, eyes, dct):
        fused, attention = self.fusion(
            self.face_branch(face),
            self.eye_branch(eyes),
            self.dct_branch(dct),
        )
        return self.classifier(fused), attention


def load_detector(bundle_dir, device="cpu"):
    bundle_dir = Path(bundle_dir)
    with open(bundle_dir / "model_config.json") as handle:
        config = json.load(handle)

    arch = config["architecture"]
    model = DeepfakeDetector(
        embed_dim=arch["embed_dim"],
        num_classes=arch["num_classes"],
        dropout=arch["dropout"],
        pretrained=False,
    )
    state = torch.load(bundle_dir / config["weights_file"], map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, config


@torch.no_grad()
def predict(model, face, eyes, dct):
    """face (N,3,256,256) and eyes (N,3,96,96) in [0,1]; dct (N,1,128,128).
    Returns (probability FAKE, shape (N,); branch gate weights, shape (N,3))."""
    device = next(model.parameters()).device
    logits, gates = model(face.to(device), eyes.to(device), dct.to(device))
    probs = torch.sigmoid(logits.float()).view(-1).cpu().numpy()
    return probs, gates.float().cpu().numpy()
