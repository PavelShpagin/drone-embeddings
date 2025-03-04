import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedTripletLoss(nn.Module):
    def __init__(self, alpha=10):
        super(WeightedTripletLoss, self).__init__()
        self.alpha = alpha

    def forward(self, query, positive, negative):
        distance_positive = (query - positive).pow(2).sum(1)
        distance_negative = (query - negative).pow(2).sum(1)
        
        # No weighting needed - each triplet contributes independently
        losses = torch.log(1 + torch.exp(self.alpha * (distance_positive - distance_negative)))
        return losses.mean()  # or .sum() depending on your preference 