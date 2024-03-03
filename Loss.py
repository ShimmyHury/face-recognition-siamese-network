import torch
import torch.nn as nn
import torch.nn.functional as functional


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, metric='Euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.metric = metric

    @staticmethod
    def calc_euclidean(x1, x2):
        return (x1 - x2).pow(2).sum(1)

    @staticmethod
    def calc_cosine(x1, x2):
        return 1 - functional.cosine_similarity(x1, x2)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        if self.metric == 'Cosine':
            distance_positive = self.calc_cosine(anchor, positive)
            distance_negative = self.calc_cosine(anchor, negative)
        else:
            distance_positive = self.calc_euclidean(anchor, positive)
            distance_negative = self.calc_euclidean(anchor, negative)

        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
