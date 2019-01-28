"""
    compute hessian vector products as well as eigenvalues of the hessian
    # copied from https://github.com/tomgoldstein/loss-landscape/blob/master/hess_vec_prod.py
"""

import torch
import time
import numpy as np
from torch import nn
from torch.autograd import Variable
from scipy.sparse.linalg import LinearOperator, eigsh
from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag

from swag.utils import flatten, unflatten_like

################################################################################
#                              Supporting Functions
################################################################################
def npvec_to_tensorlist(vec, params):
    """ Convert a numpy vector to a list of tensor with the same dimensions as params
        Args:
            vec: a 1D numpy vector
            params: a list of parameters from net
        Returns:
            rval: a list of tensors with the same shape as params
    """
    loc = 0
    rval = []
    for p in params:
        numel = p.data.numel()
        rval.append(torch.from_numpy(vec[loc:loc+numel]).view(p.data.shape).float())
        loc += numel
    assert loc == vec.size, 'The vector has more elements than the net has parameters'
    return rval


def gradtensor_to_npvec(net, include_bn=False):
    """ Extract gradients from net, and return a concatenated numpy vector.
        Args:
            net: trained model
            include_bn: If include_bn, then gradients w.r.t. BN parameters and bias
            values are also included. Otherwise only gradients with dim > 1 are considered.
        Returns:
            a concatenated numpy vector containing all gradients
    """
    filter = lambda p: include_bn or len(p.data.size()) > 1
    return np.concatenate([p.grad.data.cpu().numpy().ravel() for p in net.parameters() if filter(p)])

def gradtensor_to_tensor(net, include_bn=False):
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
    net.zero_grad() # clears grad for every parameter in the net

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = Variable(inputs), Variable(targets)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True)

        # Compute inner product of gradient with the direction vector
        #prod = Variable(torch.zeros(1)).type(type(grad_f[0].data))
        prod = torch.zeros(1, dtype=grad_f[0].dtype, device = grad_f[0].device)
        for (g, v) in zip(grad_f, vec):
            prod = prod + (g * v).sum()

        # Compute the Hessian-vector product, H*v
        # prod.backward() computes dprod/dparams for every parameter in params and
        # accumulate the gradients into the params.grad attributes
        prod.backward()

def power_method(matmul_closure, N, tolerance, max_steps, dtype, device):
    q_initial = torch.randn(N, 1, dtype = dtype, device = device)
    q_initial = q_initial / q_initial.norm()
    lambda_old = 1e10

    A_m_qcurr = matmul_closure(q_initial)
    for i in range(max_steps):
        #z_current = hess_vec_prod(q_current)
        z_current = A_m_qcurr
        q_current = z_current / z_current.norm()

        A_m_qcurr = matmul_closure(q_current)
        lambda_current = q_current.t().matmul(A_m_qcurr)
        
        if (lambda_current - lambda_old).norm() < tolerance:
            print('Convergence: ', lambda_current)
            break
        else:
            print('Step: ', i, lambda_current)
        
        lambda_old = lambda_current

    return lambda_current
################################################################################
#                  For computing Eigenvalues of Hessian
################################################################################
def min_max_hessian_eigs(net, dataloader, criterion, rank=0, use_cuda=False, verbose=False):
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
        #print(vec, vec.dtype, vec.device)
        #vec = npvec_to_tensorlist(vec, params)
        vec = unflatten_like(vec.t(), params)

        start_time = time.time()
        eval_hess_vec_prod(vec, params, net, criterion, dataloader, use_cuda)
        prod_time = time.time() - start_time
        if verbose and rank == 0: print("   Iter: %d  time: %f" % (hess_vec_prod.count, prod_time))
        #return gradtensor_to_npvec(net)
        out = gradtensor_to_tensor(net)
        return out.unsqueeze(1)

    hess_vec_prod.count = 0
    if verbose and rank == 0: print("Rank %d: computing max eigenvalue" % rank)

    #A = LinearOperator((N, N), matvec=hess_vec_prod)
    #pos_eigvals, _ = eigsh(A, k=1, tol=1e-2)
    #maxeig = power_method(hess_vec_prod, N, 1e-2, 40, device = params[0].device, dtype = params[0].dtype)
    _, pos_t_mat = lanczos_tridiag(hess_vec_prod, 100, device = params[0].device, dtype = params[0].dtype, matrix_shape=(N,N))
    pos_eigvals, _ = lanczos_tridiag_to_diag(pos_t_mat)
    print(pos_eigvals)
    # eigenvalues may not be sorted
    maxeig = torch.max(pos_eigvals)


    #maxeig = pos_eigvals[0]
    if verbose and rank == 0: print('max eigenvalue = %f' % maxeig)

    # If the largest eigenvalue is positive, shift matrix so that any negative eigenvalue is now the largest
    # We assume the smallest eigenvalue is zero or less, and so this shift is more than what we need
    #shift = maxeig*.51
    shift = 0.51 * maxeig.item()
    print(shift)
    def shifted_hess_vec_prod(vec):
        hvp = hess_vec_prod(vec)
        return -hvp + shift*vec
    
    shifted_hvp_numpy = lambda x: shifted_hess_vec_prod(torch.tensor(x).float().cuda().unsqueeze(1)).squeeze(1).cpu().numpy()


    if verbose and rank == 0: print("Rank %d: Computing shifted eigenvalue" % rank)

    """A = LinearOperator((N, N), matvec=shifted_hvp_numpy)
    neg_eigvals, _ = eigsh(A, k=1, tol=1e-2)
    mineig = neg_eigvals[0]
    print(- neg_eigvals + shift)"""
    _, neg_t_mat = lanczos_tridiag(shifted_hess_vec_prod, 200, device = params[0].device, dtype = params[0].dtype, matrix_shape=(N,N))
    neg_eigvals, _ = lanczos_tridiag_to_diag(neg_t_mat)
    mineig = torch.max(neg_eigvals)
    print(neg_eigvals)
    #mineig = neg_eigvals[0]"""
    #mineig = power_method(shifted_hess_vec_prod, N, 1e-3, 300, device = params[0].device, dtype = params[0].dtype)
    mineig = -mineig + shift
    print(mineig)
    if verbose and rank == 0: print('min eigenvalue = ' + str(mineig))

    if maxeig <= 0 and mineig > 0:
        maxeig, mineig = mineig, maxeig

    return maxeig, mineig, hess_vec_prod.count, pos_eigvals, neg_eigvals