import torch
import numpy as np
import unittest
import gpytorch

from swag.posteriors import SWAG
from swag.utils import flatten

from scipy.stats import chi2

torch.backends.cudnn.deterministic = True


class Test_SWAG_Sampling(unittest.TestCase):
    def test_swag_cov(self, **kwargs):
        model = torch.nn.Linear(300, 3, bias=True)

        swag_model = SWAG(
            torch.nn.Linear,
            in_features=300,
            out_features=3,
            bias=True,
            no_cov_mat=False,
            max_num_models=100,
            loading=False,
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        # construct swag model via training
        torch.manual_seed(0)
        for _ in range(101):
            model.zero_grad()

            input = torch.randn(100, 300)
            output = model(input)
            loss = ((torch.randn(100, 3) - output) ** 2.0).sum()
            loss.backward()

            optimizer.step()

            swag_model.collect_model(model)

        # check to ensure parameters have the correct sizes
        mean_list = []
        sq_mean_list = []
        cov_mat_sqrt_list = []
        for (module, name), param in zip(swag_model.params, model.parameters()):
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)
            cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)

            self.assertEqual(param.size(), mean.size())
            self.assertEqual(param.size(), sq_mean.size())
            self.assertEqual(
                [swag_model.max_num_models, param.numel()], list(cov_mat_sqrt.size())
            )

            mean_list.append(mean)
            sq_mean_list.append(sq_mean)
            cov_mat_sqrt_list.append(cov_mat_sqrt)

        mean = flatten(mean_list).cuda()
        sq_mean = flatten(sq_mean_list).cuda()
        cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1).cuda()

        true_cov_mat = (
            1.0 / (swag_model.max_num_models - 1)
        ) * cov_mat_sqrt.t().matmul(cov_mat_sqrt) + torch.diag(sq_mean - mean ** 2)

        test_cutoff = chi2(df=mean.numel()).ppf(
            0.95
        )  # 95% quantile of p dimensional chi-square distribution

        for scale in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
            scaled_cov_mat = true_cov_mat * scale
            scaled_cov_inv = torch.inverse(scaled_cov_mat)
            # now test to ensure that sampling has the correct covariance matrix probabilistically
            all_qforms = []
            for _ in range(2000):
                swag_model.sample(scale=scale, cov=True)
                curr_pars = []
                for (module, name) in swag_model.params:
                    curr_pars.append(getattr(module, name))
                dev = flatten(curr_pars) - mean

                # (x - mu)sigma^{-1}(x - mu)
                qform = dev.matmul(scaled_cov_inv).matmul(dev)

                all_qforms.append(qform.item())

            samples_in_cr = (np.array(all_qforms) < test_cutoff).sum()
            print(samples_in_cr)

            # between 94 and 96% of the samples should fall within the threshold
            # this should be very loose
            self.assertTrue(1880 <= samples_in_cr <= 1920)

    def test_swag_diag(self, **kwargs):
        model = torch.nn.Linear(300, 3, bias=True)

        swag_model = SWAG(
            torch.nn.Linear,
            in_features=300,
            out_features=3,
            bias=True,
            no_cov_mat=True,
            max_num_models=100,
            loading=False,
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        # construct swag model via training
        torch.manual_seed(0)
        for _ in range(101):
            model.zero_grad()

            input = torch.randn(100, 300)
            output = model(input)
            loss = ((torch.randn(100, 3) - output) ** 2.0).sum()
            loss.backward()

            optimizer.step()

            swag_model.collect_model(model)

        # check to ensure parameters have the correct sizes
        mean_list = []
        sq_mean_list = []
        for (module, name), param in zip(swag_model.params, model.parameters()):
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            self.assertEqual(param.size(), mean.size())
            self.assertEqual(param.size(), sq_mean.size())

            mean_list.append(mean)
            sq_mean_list.append(sq_mean)

        mean = flatten(mean_list).cuda()
        sq_mean = flatten(sq_mean_list).cuda()

        for scale in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
            var = scale * (sq_mean - mean ** 2)

            std = torch.sqrt(var)
            dist = torch.distributions.Normal(mean, std)

            # now test to ensure that sampling has the correct covariance matrix probabilistically
            all_qforms = 0
            for _ in range(20):
                swag_model.sample(scale=scale, cov=False)
                curr_pars = []
                for (module, name) in swag_model.params:
                    curr_pars.append(getattr(module, name))

                curr_probs = dist.cdf(flatten(curr_pars))

                # check if within 95% CI
                num_in_cr = ((curr_probs > 0.025) & (curr_probs < 0.975)).float().sum()
                # all_qforms.append( num_in_cr )
                all_qforms += num_in_cr

            # print(all_qforms/(20 * mean.numel()))
            # now compute average
            avg_prob_in_cr = all_qforms / (20 * mean.numel())

            # CLT should hold a bit tighter here
            self.assertTrue(0.945 <= avg_prob_in_cr <= 0.955)


if __name__ == "__main__":
    unittest.main()
