
import torch
import torch.nn as nn


class WeightedFocalLoss(nn.Module):


    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    @staticmethod
    def _clinical_weights(y_true: torch.Tensor) -> torch.Tensor:

        w = torch.ones_like(y_true)


        mask_low = y_true < 70
        w[mask_low] = 10.0


        mask_high = y_true > 180
        w[mask_high] = 2.0

        return w

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        se = (y_true - y_pred) ** 2
        pt =  torch.exp(-se/225)
        w = self._clinical_weights(y_true)
        focal_mod = (1.0 - pt) ** self.gamma
        loss = w * focal_mod * se
        return loss.mean()

    @staticmethod
    def to_prediction(y_pred: torch.Tensor) -> torch.Tensor:
        return y_pred.squeeze()