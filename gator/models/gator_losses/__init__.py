from dataclasses import dataclass
from typing import Literal
import torch.nn as nn
from gator.models.gator_losses.base import GatorBaseLoss

@dataclass
class GatorLossConfig:
    name: Literal["visual", "classification", "distance-based"] = "classification"

    def get_loss(self) -> type[GatorBaseLoss]:
        if self.name == "classification":
            from gator.models.gator_losses.classification import GatorClassificationLoss
            return GatorClassificationLoss
        
        elif self.name == "visual":
            from gator.models.gator_losses.visual import GatorVisualLoss
            return GatorVisualLoss
        
        elif self.name == "distance-based":
            from gator.models.gator_losses.distance_based import GatorDistanceBasedLoss
            return GatorDistanceBasedLoss