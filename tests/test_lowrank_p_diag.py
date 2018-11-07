import torch
import unittest
import gpytorch

import sys
sys.path.append('..')

from gpytorch.lazy import RootLazyTensor, DiagLazyTensor, AddedDiagLazyTensor
from gpytorch.utils import pivoted_cholesky
from gpytorch.distributions import MultivariateNormal

class Test_LowRank_P_Diag(unittest.TestCase):
    def construct_A_D(self,N=400, p=40, seed = 1):
        torch.manual_seed(seed)

        D = torch.randn(N, p)
        A = torch.randn(N).abs() * 1e-3 + 3.0

        return A, D

    def test_added_diag_lt(self,N=4000,p=40):
        A, D = self.construct_A_D(N=N,p=p)

        #this is a lazy tensor for DD'
        D_lt = RootLazyTensor(D)

        #this is a lazy tensor for diag(A)
        diag_term = DiagLazyTensor(A)

        #DD' + diag(A)
        lowrank_pdiag_lt = AddedDiagLazyTensor(diag_term, D_lt)

        #z \sim N(0,I), mean = 1
        z = torch.randn(N)
        mean = torch.ones(N)

        #N(1, DD' + diag(A))
        lazydist = MultivariateNormal(mean, lowrank_pdiag_lt)
        lazy_lprob = lazydist.log_prob(z)

        exact_dist = torch.distributions.MultivariateNormal(mean, lowrank_pdiag_lt.evaluate())
        exact_lprob = exact_dist.log_prob(z)

        rel_error = torch.norm( lazy_lprob - exact_lprob ) / exact_lprob.norm()

        self.assertLess(rel_error.cpu().item(), 0.01)




if __name__ == "__main__":
    unittest.main()
