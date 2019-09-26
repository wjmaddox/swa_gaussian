import torch
import unittest
import gpytorch

from gpytorch.lazy import RootLazyTensor, DiagLazyTensor, AddedDiagLazyTensor
from gpytorch.distributions import MultivariateNormal

torch.backends.cudnn.deterministic = True


class Test_LowRank_P_Diag(unittest.TestCase):
    def test_added_diag_lt(self, N=10000, p=20, use_cuda=False, seed=1):

        torch.manual_seed(seed)

        if torch.cuda.is_available() and use_cuda:
            print("Using cuda")
            device = torch.device("cuda")
            torch.cuda.manual_seed_all(seed)
        else:
            device = torch.device("cpu")

        D = torch.randn(N, p, device=device)
        A = torch.randn(N, device=device).abs() * 1e-3 + 0.1

        # this is a lazy tensor for DD'
        D_lt = RootLazyTensor(D)

        # this is a lazy tensor for diag(A)
        diag_term = DiagLazyTensor(A)

        # DD' + diag(A)
        lowrank_pdiag_lt = AddedDiagLazyTensor(diag_term, D_lt)

        # z \sim N(0,I), mean = 1
        z = torch.randn(N, device=device)
        mean = torch.ones(N, device=device)

        diff = mean - z

        print(lowrank_pdiag_lt.log_det())
        logdet = lowrank_pdiag_lt.log_det()
        inv_matmul = lowrank_pdiag_lt.inv_matmul(diff.unsqueeze(1)).squeeze(1)
        inv_matmul_quad = torch.dot(diff, inv_matmul)

        """inv_matmul_quad_qld, logdet_qld = lowrank_pdiag_lt.inv_quad_log_det(inv_quad_rhs=diff.unsqueeze(1), log_det = True)
        
        """

        """from gpytorch.functions._inv_quad_log_det import InvQuadLogDet
        iqld_construct = InvQuadLogDet(gpytorch.lazy.lazy_tensor_representation_tree.LazyTensorRepresentationTree(lowrank_pdiag_lt),
                            matrix_shape=lowrank_pdiag_lt.matrix_shape,
                            dtype=lowrank_pdiag_lt.dtype,
                            device=lowrank_pdiag_lt.device,
                            inv_quad=True,
                            log_det=True,
                            preconditioner=lowrank_pdiag_lt._preconditioner()[0],
                            log_det_correction=lowrank_pdiag_lt._preconditioner()[1])
        inv_matmul_quad_qld, logdet_qld = iqld_construct(diff.unsqueeze(1))"""
        num_random_probes = gpytorch.settings.num_trace_samples.value()
        probe_vectors = torch.empty(
            lowrank_pdiag_lt.matrix_shape[-1],
            num_random_probes,
            dtype=lowrank_pdiag_lt.dtype,
            device=lowrank_pdiag_lt.device,
        )
        probe_vectors.bernoulli_().mul_(2).add_(-1)
        probe_vector_norms = torch.norm(probe_vectors, 2, dim=-2, keepdim=True)
        probe_vectors = probe_vectors.div(probe_vector_norms)

        # diff_norm = diff.norm()
        # diff = diff/diff_norm
        rhs = torch.cat([diff.unsqueeze(1), probe_vectors], dim=1)

        solves, t_mat = gpytorch.utils.linear_cg(
            lowrank_pdiag_lt.matmul,
            rhs,
            n_tridiag=num_random_probes,
            max_iter=gpytorch.settings.max_cg_iterations.value(),
            max_tridiag_iter=gpytorch.settings.max_lanczos_quadrature_iterations.value(),
            preconditioner=lowrank_pdiag_lt._preconditioner()[0],
        )
        # print(solves)
        inv_matmul_qld = solves[:, 0]  # * diff_norm

        diff_solve = gpytorch.utils.linear_cg(
            lowrank_pdiag_lt.matmul,
            diff.unsqueeze(1),
            max_iter=gpytorch.settings.max_cg_iterations.value(),
            preconditioner=lowrank_pdiag_lt._preconditioner()[0],
        )
        print("diff_solve_norm: ", diff_solve.norm())
        print(
            "diff between multiple linear_cg: ",
            (inv_matmul_qld.unsqueeze(1) - diff_solve).norm() / diff_solve.norm(),
        )

        eigenvalues, eigenvectors = gpytorch.utils.lanczos.lanczos_tridiag_to_diag(
            t_mat
        )
        slq = gpytorch.utils.StochasticLQ()
        log_det_term, = slq.evaluate(
            lowrank_pdiag_lt.matrix_shape,
            eigenvalues,
            eigenvectors,
            [lambda x: x.log()],
        )
        logdet_qld = log_det_term + lowrank_pdiag_lt._preconditioner()[1]

        print("Log det difference: ", (logdet - logdet_qld).norm() / logdet.norm())
        print(
            "inv matmul difference: ",
            (inv_matmul - inv_matmul_qld).norm() / inv_matmul_quad.norm(),
        )

        # N(1, DD' + diag(A))
        lazydist = MultivariateNormal(mean, lowrank_pdiag_lt)
        lazy_lprob = lazydist.log_prob(z)

        # exact log probability with Cholesky decomposition
        exact_dist = torch.distributions.MultivariateNormal(
            mean, lowrank_pdiag_lt.evaluate().float()
        )
        exact_lprob = exact_dist.log_prob(z)

        print(lazy_lprob, exact_lprob)
        rel_error = torch.norm(lazy_lprob - exact_lprob) / exact_lprob.norm()

        self.assertLess(rel_error.cpu().item(), 0.01)


if __name__ == "__main__":
    unittest.main()
