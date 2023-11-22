import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

class LogitAdjust(nn.Module):
    """Logit adjusted loss: https://arxiv.org/pdf/2007.07314.pdf
        This is a PyTorch module for adjusting logits based on class frequencies and cross-entropy loss."""
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)

class FocalLoss(nn.modules.loss._WeightedLoss):
    # The FocalLoss class implements a custom loss function for classification tasks that balances class
    # weights using the alpha parameter and applies a focal loss function to penalize misclassifications.
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, x, target):

        ce_loss = F.cross_entropy(x, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class FocalLC(nn.Module):
    def __init__(self, cls_num_list, tau=1, gamma=2, weight=None,reduction='mean'):
        super(FocalLC, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, x, target):
        x_m = x + self.m_list #logit adjust
        ce_loss = F.cross_entropy(x_m, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def linear_combination(x, y, epsilon):
    """
    The function returns a linear combination of two input values with a specified weight.
    
    :param x: The first input value for the linear combination function
    :param y: The parameter "y" in the function represents one of the two values that we want to combine
    using a linear combination
    :param epsilon: Epsilon is a scalar value between 0 and 1 that determines the weight given to the
    first input x in the linear combination. Specifically, epsilon determines the proportion of x in the
    final output, with (1-epsilon) determining the proportion of y
    :return: the linear combination of two input values x and y, where the weight of x is determined by
    the value of epsilon. Specifically, it returns the value of epsilon times x plus (1-epsilon) times
    y.
    """
    return epsilon * x + (1 - epsilon) * y

def reduce_loss(loss, reduction='mean'):
    """
    The function reduces the loss by either taking the mean or sum of the loss values based on the
    specified reduction method.
    
    :param loss: The loss is a tensor that represents the difference between the predicted output and
    the actual output of a machine learning model. It is a measure of how well the model is performing
    on a given task
    :param reduction: The reduction parameter specifies how the loss should be reduced. It can take one
    of three values: 'mean', 'sum', or None. If reduction is 'mean', the loss is averaged over all
    elements in the tensor. If reduction is 'sum', the loss is summed over all elements in the, defaults
    to mean (optional)
    :return: The function `reduce_loss` returns the mean of the input `loss` if `reduction` is set to
    `'mean'`, the sum of the input `loss` if `reduction` is set to `'sum'`, and the input `loss` itself
    if `reduction` is any other value.
    """
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

class EQLv2(nn.Module):
    """Equalization Loss v2: https://arxiv.org/abs/2012.08548.
        adapted from: https://github.com/tztztztztz/eqlv2/blob/master/mmdet/models/losses/eqlv2.py#L90 """
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=7,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 vis_grad=False,
                 test_with_obj=True):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True

        # cfg for eqlv2
        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        # At the beginning of training, we set a high value (eg. 100)
        # for the initial gradient ratio so that the weight for pos gradients and neg gradients are 1.
        self.register_buffer('pos_neg', torch.ones(self.num_classes) * 100)

        self.test_with_obj = test_with_obj

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes.type(torch.int64)] = 1
            return target

        target = expand_label(cls_score, label)

        pos_w, neg_w = self.get_weight(cls_score)

        pos_w = pos_w.to(target.device)
        neg_w = neg_w.to(target.device)

        weight = pos_w * target + neg_w * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,
                                                      reduction='none')
        cls_loss = torch.sum(cls_loss * weight) / self.n_i

        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

        return self.loss_weight * cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def get_activation(self, cls_score):
        cls_score = torch.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        if self.test_with_obj:
            cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        # do not collect grad for objectiveness branch [:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0)
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)

        pos_grad = pos_grad.to(self.pos_grad.device)
        neg_grad = neg_grad.to(self.neg_grad.device)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)

    def get_weight(self, cls_score):
        neg_w = self.map_func(self.pos_neg)
        pos_w = 1 + self.alpha * (1 - neg_w)
        neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
        pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w

class LabelSmoothingCrossEntropy(nn.Module):
# This is a PyTorch module for calculating label smoothing cross entropy loss.
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, x, target):
        n = x.size()[-1]
        log_preds = F.log_softmax(x, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)