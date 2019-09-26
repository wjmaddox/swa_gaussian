"""
    utils for gradient covariance experiment
"""

import itertools
import tqdm
import torch


def initialize_grad_list(model):
    grad_list = []
    for param in model.parameters():
        param_dict = {
            "sum": torch.zeros_like(param, requires_grad=False),
            "sq_sum": torch.zeros_like(param, requires_grad=False),
            "num_models": 0,
        }
        grad_list.append(param_dict)

    return grad_list


def collect_grads(model, model_grads_list):
    for (param, grad_dict) in zip(model.parameters(), model_grads_list):
        grad_dict["sum"] += param.grad.data
        grad_dict["sq_sum"] += param.grad.data ** 2.0
        grad_dict["num_models"] += 1


def compute_opt_lr(grad_list, momentum, dataset_size):

    var_diag_sum = 0
    num_params = 0

    if grad_list[0]["num_models"] < 2:
        print("No models stored yet")
        return None, None

    for grad_dict in grad_list:
        # (1/n sum x_i )^2
        first_moment_squared = (grad_dict["sum"] / grad_dict["num_models"]) ** 2

        # 1/n sum x_i^2
        second_moment = grad_dict["sq_sum"] / grad_dict["num_models"]

        # E(x^2) - E(x)^2
        var = second_moment - first_moment_squared

        var_diag_sum += var.sum()

        num_params += var.numel()

    # this is the noise up-scaled by a factor of S, so this cancels with the upper batch size
    grad_noise = var_diag_sum.item()
    # optimal lr is 2 * \mu * S /N * D/tr(C)

    return 2 * num_params / (dataset_size * grad_noise), grad_noise


def train_epoch(
    loader,
    model,
    criterion,
    optimizer,
    model_grads_list=None,
    cuda=True,
    regression=False,
    verbose=False,
    subset=None,
):
    loss_sum = 0.0
    correct = 0.0
    verb_stage = 0

    num_objects_current = 0
    num_batches = len(loader)

    model.train()

    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    dataset_size = len(loader.dataset)

    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches)

    for i, (input, target) in enumerate(loader):
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        loss, output = criterion(model, input, target)

        optimizer.zero_grad()
        loss.backward()

        if model_grads_list is not None:
            # store gradients
            collect_grads(model, model_grads_list)

            # compute optimal learning rates
            # note that \mu = 1 - sgd['momentum'] bc of differences in pytorch's implementation
            lr, grad_noise = compute_opt_lr(
                model_grads_list,
                1 - optimizer.param_groups[0]["momentum"],
                dataset_size,
            )

        optimizer.step()

        loss_sum += loss.data.item() * input.size(0)

        if not regression:
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        num_objects_current += input.size(0)

        if verbose and 10 * (i + 1) / num_batches >= verb_stage + 1:
            print(
                "Stage %d/10. Loss: %12.4f. Acc: %6.2f"
                % (
                    verb_stage + 1,
                    loss_sum / num_objects_current,
                    correct / num_objects_current * 100.0,
                )
            )
            if model_grads_list is not None:
                print("Learning Rate: %.3f. tr(V(\hat(g))): %.3f" % (lr, grad_noise))
            verb_stage += 1

    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": None if regression else correct / num_objects_current * 100.0,
    }
