import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_accuracy(labels, outputs):
    logits = torch.max(outputs, 1)[1].data
    one_hots = torch.zeros(*labels.size())
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    score_sum = scores.sum()
    score = score_sum/scores.size(0)
    return score.numpy()


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * targets * inputs - self.gamma * torch.log(1 +
                            torch.exp(-1.0 * inputs)))

        loss = modulator * BCE_loss

        weighted_loss = self.alpha * loss

        focal_loss = torch.mean(weighted_loss)

        return focal_loss


class CBLoss(nn.Module):
    def __init__(self, gamma=0.5, beta=0.9999, logits=True, reduce=True, eff_num=True):
        super(CBLoss, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.logits = logits
        self.reduce = reduce
        self.samples_per_cls = np.array([296., 2994., 609., 1639., 10332., 1425., 7874., 7837., 97., 21.])
        self.len = 11246
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.eff_num = eff_num

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * targets * inputs - self.gamma * torch.log(1 +
                            torch.exp(-1.0 * inputs)))

        loss = modulator * BCE_loss

        weights = (self.len - self.samples_per_cls)/self.samples_per_cls
        weights = torch.tensor(weights).float().to(self.device)
        weights = weights.repeat(targets.size(0), 1)

        alpha = targets * weights + (1-targets) * (1/weights)

        if self.eff_num:
            effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * 10
            weights = torch.tensor(weights).float().to(self.device)
            weights = weights.unsqueeze(0)
            weights = weights.repeat(targets.shape[0], 1) * targets
            weights = weights.sum(1)
            weights = weights.unsqueeze(1)
            weights = weights.repeat(1, 10)

        weighted_loss = alpha * loss
        cbloss = torch.sum(weighted_loss)
        cbloss /= weighted_loss.size(0)

        return cbloss
