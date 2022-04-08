import torch
import numpy as np

def bpr_loss(positive_score, negative_score):
    """
    Bayesian Personalised Ranking loss
    Args:
        positive_score: (tensor<float>) predicted scores for known positive items
        negative_score: (tensor<float>) predicted scores for negative sample items
    Returns:
        loss: (float) the mean value of the summed loss
    """
    eps = 1e-7 # Smooth the argument of the log to prevent potential numerical underflows
    loss = -torch.log(torch.sigmoid(positive_score - negative_score) + eps)
    return loss.mean()

def ce_loss(score, label):
    """
    Cross-entropy loss
    Args:
        score: (tensor<float>) predicted scores for items
        label: (tensor<float> or float) item labels (1 for positive and 0 for negative samples)
    Returns:
        loss: (float) the mean value of the summed loss
    """
    eps = 1e-7 # Smooth the argument of the log to prevent potential numerical underflows
    loss = -label * torch.log(torch.sigmoid(score) + eps) \
           -(1.0 - label) * torch.log(1.0 - torch.sigmoid(score) + eps)
    return loss.mean()

def sample_items(num_items, shape):
    """
    Randomly sample a number of items
    Args
        num_items: (int) total number of items from which we should sample: the maximum value of a sampled item id will be
        smaller than this.
        shape: (int or tuple<int>) shape of the sampled array.
    Returns
        items: (array<int>) sampled item ids.
    """

    res_items = np.random.randint(0, num_items, shape, dtype=np.int64)
    return res_items