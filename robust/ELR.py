import torch.nn as nn
import torch
import torch.nn.functional as F


class ELRLoss(nn.Module):
    def __init__(self, num_examp, num_classes=10, e_lambda=3, beta=0.7, device=None):
        r"""Early Learning Regularization.
        Parameters
        * `num_examp` Total number of training examples.
        * `num_classes` Number of classes in the classification problem.
        * `lambda` Regularization strength; must be a positive float, controling the strength of the ELR.
        * `beta` Temporal ensembling momentum for target estimation.
        """
        super(ELRLoss, self).__init__()
        self.num_classes = num_classes
        self.device = 'cpu' if device is None else device
        self.target = torch.zeros(num_examp, self.num_classes).to(device)
        self.beta = beta
        self.e_lambda = e_lambda

    def forward(self, output, label, index):
        r"""Early Learning Regularization.
         Args
         * `index` Training sample index, due to training set shuffling, index is used to track training examples in
         fifferent iterations.
         * `output` Model's logits, same as PyTorch provided loss functions.
         * `label` Labels, same as PyTorch provided loss functions.
         """

        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1 - self.beta) * ((y_pred_)/(y_pred_).sum(dim=1, keepdim=True))
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1 - (self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + self.e_lambda * elr_reg
        return final_loss
