"""
    compute hessian vector products as well as eigenvalues of the hessian
    # copied from https://github.com/tomgoldstein/loss-landscape/blob/master/hess_vec_prod.py
    # code re-written to use gpu by default and then to use gpytorch
"""

import torch
import time
import numpy as np
from torch import nn
from torch.autograd import Variable

from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag

from swag.utils import flatten, unflatten_like

################################################################################
#                              Supporting Functions
################################################################################
def gradtensor_to_tensor(net, include_bn=False):
    """
        convert the grad tensors to a list
    """
    filter = lambda p: include_bn or len(p.data.size()) > 1
    return flatten([p.grad.data for p in net.parameters() if filter(p)])


################################################################################
#                  For computing Hessian-vector products
################################################################################
def eval_hess_vec_prod(vec, params, net, criterion, dataloader, use_cuda=False):
    """
    Evaluate product of the Hessian of the loss function with a direction vector "vec".
    The product result is saved in the grad of net.
    Args:
        vec: a list of tensor with the same dimensions as "params".
        params: the parameter list of the net (ignoring biases and BN parameters).
        net: model with trained parameters.
        criterion: loss function.
        dataloader: dataloader for the dataset.
        use_cuda: use GPU.
    """

    if use_cuda:
        net.cuda()
        vec = [v.cuda() for v in vec]

    net.eval()
    net.zero_grad()  # clears grad for every parameter in the net

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = Variable(inputs), Variable(targets)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True)

        # Compute inner product of gradient with the direction vector
        # prod = Variable(torch.zeros(1)).type(type(grad_f[0].data))
        prod = torch.zeros(1, dtype=grad_f[0].dtype, device=grad_f[0].device)
        for (g, v) in zip(grad_f, vec):
            prod = prod + (g * v).sum()

        # Compute the Hessian-vector product, H*v
        # prod.backward() computes dprod/dparams for every parameter in params and
        # accumulate the gradients into the params.grad attributes
        prod.backward()


################################################################################
#                  For computing Eigenvalues of Hessian
################################################################################
def min_max_hessian_eigs(
    net, dataloader, criterion, rank=0, use_cuda=False, verbose=False
):
    """
        Compute the largest and the smallest eigenvalues of the Hessian marix.
        Args:
            net: the trained model.
            dataloader: dataloader for the dataset, may use a subset of it.
            criterion: loss function.
            rank: rank of the working node.
            use_cuda: use GPU
            verbose: print more information
        Returns:
            maxeig: max eigenvalue
            mineig: min eigenvalue
            hess_vec_prod.count: number of iterations for calculating max and min eigenvalues
    """

    params = [p for p in net.parameters() if len(p.size()) > 1]
    N = sum(p.numel() for p in params)

    def hess_vec_prod(vec):
        hess_vec_prod.count += 1  # simulates a static variable
        vec = unflatten_like(vec.t(), params)

        start_time = time.time()
        eval_hess_vec_prod(vec, params, net, criterion, dataloader, use_cuda)
        prod_time = time.time() - start_time
        if verbose and rank == 0:
            print("   Iter: %d  time: %f" % (hess_vec_prod.count, prod_time))
        out = gradtensor_to_tensor(net)
        return out.unsqueeze(1)

    hess_vec_prod.count = 0
    if verbose and rank == 0:
        print("Rank %d: computing max eigenvalue" % rank)

    # use lanczos to get the t and q matrices out
    _, pos_t_mat = lanczos_tridiag(
        hess_vec_prod,
        100,
        device=params[0].device,
        dtype=params[0].dtype,
        matrix_shape=(N, N),
    )
    # convert the tridiagonal t matrix to the eigenvalues
    pos_eigvals, _ = lanczos_tridiag_to_diag(pos_t_mat)
    print(pos_eigvals)
    # eigenvalues may not be sorted
    maxeig = torch.max(pos_eigvals)

    # maxeig = pos_eigvals[0]
    if verbose and rank == 0:
        print("max eigenvalue = %f" % maxeig)

    # If the largest eigenvalue is positive, shift matrix so that any negative eigenvalue is now the largest
    # We assume the smallest eigenvalue is zero or less, and so this shift is more than what we need
    # shift = maxeig*.51
    shift = 0.51 * maxeig.item()
    print(shift)

    def shifted_hess_vec_prod(vec):
        hvp = hess_vec_prod(vec)
        return -hvp + shift * vec

    if verbose and rank == 0:
        print("Rank %d: Computing shifted eigenvalue" % rank)

    # now run lanczos on the shifted eigenvalues
    _, neg_t_mat = lanczos_tridiag(
        shifted_hess_vec_prod,
        200,
        device=params[0].device,
        dtype=params[0].dtype,
        matrix_shape=(N, N),
    )
    neg_eigvals, _ = lanczos_tridiag_to_diag(neg_t_mat)
    mineig = torch.max(neg_eigvals)
    print(neg_eigvals)

    mineig = -mineig + shift
    print(mineig)
    if verbose and rank == 0:
        print("min eigenvalue = " + str(mineig))

    if maxeig <= 0 and mineig > 0:
        maxeig, mineig = mineig, maxeig

    return maxeig, mineig, hess_vec_prod.count, pos_eigvals, neg_eigvals
