# The code here is based on the code at
# https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np


def logits_from_probs(prob_arr):
    return np.log(prob_arr)


def optimal_temp_scale(probs_arr, labels_arr, lr=0.01, max_iter=50):
    probs = torch.from_numpy(probs_arr).float()
    labels = torch.from_numpy(labels_arr.astype(int))
    logits = torch.log(probs + 1e-12)
    nll_criterion = nn.CrossEntropyLoss()

    before_temperature_nll = nll_criterion(logits, labels).item()
    print("Before temperature - NLL: %.3f" % (before_temperature_nll))

    T = Variable(torch.ones(1), requires_grad=True)

    optimizer = optim.LBFGS([T], lr=lr, max_iter=max_iter)

    def eval():
        loss = nll_criterion(logits / T, labels)
        loss.backward(retain_graph=True)
        return loss

    optimizer.step(eval)

    after_temperature_nll = nll_criterion(logits / T, labels).item()
    print(
        "After temperature - NLL: %.3f" % (after_temperature_nll), ", Temperature:", T
    )

    return T.item(), F.softmax(logits / T).data.numpy()


def rescale_temp(probs_arr, temp):
    logits = np.log(probs_arr)
    logits /= temp
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=1)[:, None]
    return probs
