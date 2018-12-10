import torch
import unittest
import gpytorch

from gpytorch.lazy import RootLazyTensor, DiagLazyTensor, AddedDiagLazyTensor
from gpytorch.distributions import MultivariateNormal

torch.backends.cudnn.deterministic = True

def run_linear_cg_solve_on_device(device, N = 50000, p = 20):
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    D = torch.randn(N, p).to(device)
    A = torch.randn(N).abs().to(device) * 1e-3 + 0.1
    #print('mean of a: ', A.mean(), torch.var(A).sqrt())
    #D = D.double()
    #A = A.double()

    #this is a lazy tensor for DD'
    D_lt = RootLazyTensor(D)

    #this is a lazy tensor for diag(A)
    diag_term = DiagLazyTensor(A)

    #DD' + diag(A)
    lowrank_pdiag_lt = AddedDiagLazyTensor(diag_term, D_lt)

    #generate vector that we actually care about and associated probe vectors
    diff = torch.ones(N, 1, device = device)#.double()
    probe_vectors = torch.randn(diff.size(0), 2, device = device)
    #diff_norm = diff.norm()
    diff = diff#/diff_norm

    rhs = torch.cat([diff, probe_vectors],dim=1)
    print(rhs.size())
    max_iter = 3
    #lowrank_pdiag_lt = lowrank_pdiag_lt.evaluate().contiguous().cpu().to(device)

    solves = gpytorch.utils.linear_cg(
        lowrank_pdiag_lt.matmul,
        rhs,
        max_iter=max_iter,
        max_tridiag_iter=0,
        preconditioner=None
    )
    #print(solves.size(), rhs.size())
    inv_matmul_qld = solves[:,0].unsqueeze(1)

    diff_solve = gpytorch.utils.linear_cg(
        lowrank_pdiag_lt.matmul,
        diff,
        max_iter=max_iter,
        max_tridiag_iter=0,
        preconditioner=None
    )
    #print('diff_solve_norm: ', diff_solve.norm())
    print('size of solves: ', inv_matmul_qld.size(), diff_solve.size())
    print(device, 'diff between multiple linear_cg: ', (inv_matmul_qld - diff_solve).norm()/diff_solve.norm())

    return inv_matmul_qld, diff_solve, diff, D, A

gpu_multiple_im, gpu_single_im, gpu_diff, gpu_D, gpu_A = run_linear_cg_solve_on_device(torch.device('cuda'))
cpu_multiple_im, cpu_single_im, cpu_diff, cpu_D, cpu_A = run_linear_cg_solve_on_device(torch.device('cpu'))

gpu_multiple_im, gpu_single_im = gpu_multiple_im.cpu(), gpu_single_im.cpu()

multiple_im_diff = (cpu_multiple_im - gpu_multiple_im).norm()/cpu_multiple_im.norm()
single_im_diff = (cpu_single_im - gpu_single_im).norm()/cpu_single_im.norm()

print('Sanity check (diff): ', (gpu_diff.cpu() - cpu_diff).norm())
print('Sanity check (D):', (gpu_D.cpu() - cpu_D).norm())
print('Sanity check (A): ', (gpu_A.cpu() - cpu_A).norm())
print('Multiple inverse matmul sizes: ', cpu_multiple_im.size(), gpu_multiple_im.size())
print('Single inverse matmul sizes: ', cpu_single_im.size(), gpu_single_im.size())
print('Multiple Inverse matmul difference: ', multiple_im_diff)
print('Single Inverse matmul difference: ', single_im_diff)

