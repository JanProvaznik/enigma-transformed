from dataclasses import dataclass
from typing import Type, Optional
from src.ByT5Dataset import ByT5Dataset
from scipy.spatial.distance import jensenshannon

def js_divergence(dict1, dict2):
    """Calculate the Jensen-Shannon divergence between two dictionaries/counters.
    """
    # Get the union of keys from both dictionaries
    keys = set(dict1.keys()).union(set(dict2.keys()))
    
    # Convert frequency dictionaries to probability distributions
    total1 = sum(dict1.values())
    total2 = sum(dict2.values())
    prob_dist1 = [dict1.get(key, 0) / total1 for key in keys]
    prob_dist2 = [dict2.get(key, 0) / total2 for key in keys]
    
    # Calculate Jensen-Shannon divergence
    divergence = jensenshannon(prob_dist1, prob_dist2)
    
    return divergence

@dataclass
class Model:
    dataset_class: Type[ByT5Dataset]
    name: str
    language: str
    slurm_id: int
    is_checkpoint: bool = False
    checkpoint_step: int = 0
    noise_proportion: Optional[float] = None

    def path(self):
        if self.is_checkpoint:
            return f"./logs/{self.slurm_id}/output/checkpoint-{self.checkpoint_step}"
        else:
            return f"./logs/{self.slurm_id}/model"