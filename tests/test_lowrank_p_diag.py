import torch
import unittest
import gpytorch

import sys
sys.path.append('..')

from gpytorch.lazy import RootLazyTensor, DiagLazyTensor, AddedDiagLazyTensor
from gpytorch.utils import pivoted_cholesky

class Test_LowRank_P_Diag(unittest.TestCase):
    def construct_A_D(self,N=400, p=40, seed = 1):
        torch.manual_seed(seed)

        D = torch.randn(N, p)
        A = torch.randn(N).abs() * 1e-3 + 3.0

        return A, D

    def test_added_diag_lt(self,N=400,p=40):
        A, D = self.construct_A_D(N=N,p=p)

        D_lt = RootLazyTensor(D)

        diag_term = DiagLazyTensor(A)

        lowrank_pdiag_lt = AddedDiagLazyTensor(diag_term, D_lt)

        pivoted_l = pivoted_cholesky.pivoted_cholesky(lowrank_pdiag_lt, 100)

        pivoted_llt = pivoted_l.t().matmul(pivoted_l)
        print(pivoted_llt.size())
        approx_error = torch.norm( pivoted_llt - lowrank_pdiag_lt.evaluate() ) / lowrank_pdiag_lt.evaluate().norm() 
        print(approx_error)
        #z = torch.randn(N)

        #pchol_rsample = pivoted_l.matmul(z)

        #true_chol = torch.potrf(lowrank_pdiag_lt.evaluate(), upper = False)
        #print(true_chol.size(), pivoted_l.size())
        #true_rsample = true_chol.matmul(z)

        #print( (pchol_rsample - true_rsample).norm() / true_rsample.norm() )


if __name__ == "__main__":
    unittest.main()
