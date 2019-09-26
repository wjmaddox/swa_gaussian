"""
    training helpers for segmentation
    ported from: https://github.com/bfortuner/pytorch_tiramisu
"""
import os
import sys
import math
import string
import random
import shutil
import numpy as np

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

from . import imgs as img_utils


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()


def predict(model, input_loader, n_batches=1):
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    for input, target in input_loader:
        data = Variable(input.cuda(), volatile=True)
        label = Variable(target.cuda())
        output = model(data)
        pred = get_predictions(output)
        predictions.append([input, target, pred])
    return predictions


def view_sample_predictions(model, loader, n):
    inputs, targets = next(iter(loader))
    data = Variable(inputs.cuda(), volatile=True)
    label = Variable(targets.cuda())
    output = model(data)
    pred = get_predictions(output)
    batch_size = inputs.size(0)
    for i in range(min(n, batch_size)):
        img_utils.view_image(inputs[i])
        img_utils.view_annotated(targets[i])
        img_utils.view_annotated(pred[i])


def get_predictions(output_batch):
    bs, c, h, w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs, h, w)
    return indices


def train(model, trn_loader, optimizer, criterion):
    model.train()
    trn_loss = 0
    trn_error = 0
    for idx, (inputs, targets) in enumerate(trn_loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        optimizer.zero_grad()
        loss_dict = criterion(model, inputs, targets)
        loss, output = loss_dict["loss"], loss_dict["output"]

        loss.backward()
        optimizer.step()

        trn_loss += loss.item()

        _, _, trn_acc_curr = numpy_metrics(
            output.data.cpu().numpy(), targets.data.cpu().numpy()
        )
        trn_error += 1 - trn_acc_curr

    trn_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    return trn_loss, trn_error


def test(
    model,
    test_loader,
    criterion,
    num_classes=11,
    return_outputs=False,
    return_scale=False,
):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_error = 0
        I_tot = np.zeros(num_classes)
        U_tot = np.zeros(num_classes)

        if return_outputs:
            output_list = []
            target_list = []

            scale_list = []

        for data, target in test_loader:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(data)

            loss_dict = criterion(model, data, target)
            loss, output = loss_dict["loss"], loss_dict["output"]
            # test_loss += masked_loss(output, target, criterion)
            test_loss += loss

            I, U, acc = numpy_metrics(
                output.cpu().numpy(),
                target.cpu().numpy(),
                n_classes=11,
                void_labels=[11],
            )
            I_tot += I
            U_tot += U
            test_error += 1 - acc

            if return_outputs:
                output_list.append(output.cpu().numpy())
                target_list.append(target.cpu().numpy())

            if return_scale:
                scale_list.append(loss_dict["scale"].cpu().numpy())

        test_loss /= len(test_loader)
        test_error /= len(test_loader)
        m_jacc = np.mean(I_tot / U_tot)

        if not return_outputs:
            return test_loss, test_error, m_jacc
        else:
            return (
                test_loss,
                test_error,
                m_jacc,
                {"outputs": output_list, "targets": target_list, "scales": scale_list},
            )


def numpy_metrics(y_pred, y_true, n_classes=11, void_labels=[11]):
    """
    Similar to theano_metrics to metrics but instead y_pred and y_true are now numpy arrays
    from: https://github.com/SimJeg/FC-DenseNet/blob/master/metrics.py
    void label is 11 by default
    """

    # Put y_pred and y_true under the same shape
    y_pred = np.argmax(y_pred, axis=1)

    # We use not_void in case the prediction falls in the void class of the groundtruth
    not_void = ~np.any([y_true == label for label in void_labels], axis=0)

    I = np.zeros(n_classes)
    U = np.zeros(n_classes)

    for i in range(n_classes):
        y_true_i = y_true == i
        y_pred_i = y_pred == i

        I[i] = np.sum(y_true_i & y_pred_i)
        U[i] = np.sum((y_true_i | y_pred_i) & not_void)

    accuracy = np.sum(I) / np.sum(not_void)
    return I, U, accuracy
