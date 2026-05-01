from dataclasses import dataclass
from typing import Literal
from gator.models.gator_visualizer.base import GatorBaseVis

@dataclass
class GatorVisConfig:
    name: Literal["visual", "classification", "distance-based", "mae"] = "classification"

    def get_visualizer(self) -> type[GatorBaseVis]:
        if self.name == "classification":
            from gator.models.gator_visualizer.classification import ClassificationVis
            return ClassificationVis
        
        elif self.name == "visual":
            from gator.models.gator_visualizer.visual import VisualVis
            return VisualVis
        
        elif self.name == "distance-based":
            from gator.models.gator_visualizer.distance_based import DistanceBasedVis
            return DistanceBasedVis
        
        elif self.name == "mae":
            from gator.models.gator_visualizer.mae import MAEVis
            return MAEVis
