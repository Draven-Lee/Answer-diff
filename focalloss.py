import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_accuracy(labels, outputs):
    logits = torch.max(outputs, 1)[1].data
    one_hots = torch.zeros(*labels.size())
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    score_sum = scores.sum()
    score = score_sum/scores.size(0)
    return score.numpy()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)**32
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss