from dataclasses import dataclass
from functools import partial
from typing import Literal
import torch.nn as nn
from gator.models.gator_losses.base import GatorBaseLoss

@dataclass
class GatorLossConfig:
    name: Literal["visual", "classification", "distance-based", "similarity"] = "classification"
    tau: float = 0.01
    """
    The `tau` is only used for similarity loss and it controls how much the
    similarity between patches affects the position probability.
    """

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
        
        elif self.name == "similarity":
            from gator.models.gator_losses.similarity import GatorSimilarityLoss
            return partial(GatorSimilarityLoss, tau=self.tau)