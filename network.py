import torch
import torch.nn as nn
from torchvision import models

class MicrostructureNet(nn.Module):
    def __init__(self, lbp_settings=None, freeze_backbone=True):
        """
        lbp_params: 
        """
        super().__init__()
        self.lbp_settings = lbp_settings
        self.len_lbp_config = len(lbp_settings)

        # ---------------- Backbone ----------------
        backbone = models.resnet18(weights=None)
        self.rgb_extractor = nn.Sequential(*list(backbone.children())[:-1])
        rgb_dim = backbone.fc.in_features

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.rgb_extractor.parameters():
                param.requires_grad = False

        # ---------------- LBP branch ----------------
        if lbp_settings:
            self.lbp_branch = nn.Sequential(
                nn.Conv2d(self.len_lbp_config, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            lbp_dim = 32
        else:
            lbp_dim = 0

        # ---------------- Output head ----------------
        self.head = nn.Sequential(
            nn.Linear(rgb_dim + lbp_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)  # UB vs LB/M
        )

    def forward(self, rgb, lbp=None):
        # Extract RGB features
        x_rgb = self.rgb_extractor(rgb).flatten(1)

        # Extract LBP features
        if self.len_lbp_config:
            if lbp is None:
                raise ValueError("LBP input required when len_lbp_config > 0")

            # If LBP is a list of tensors, concatenate along channel dim
            if isinstance(lbp, list):
                lbp = torch.cat(lbp, dim=1)

            x_lbp = self.lbp_branch(lbp).flatten(1)
            x = torch.cat([x_rgb, x_lbp], dim=1)
        else:
            x = x_rgb

        return self.head(x)
