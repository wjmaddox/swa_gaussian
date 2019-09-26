import torch
import numpy as np
import torch.distributions
import time

from ..utils import eval


def laplace_parameters(module, params, no_cov_mat=True, max_num_models=0):
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            print(module, name)
            continue
        data = module._parameters[name].data
        module._parameters.pop(name)
        module.register_buffer("%s_mean" % name, data.new(data.size()).zero_())
        module.register_buffer("%s_var" % name, data.new(data.size()).zero_())
        module.register_buffer(name, data.new(data.size()).zero_())

        if no_cov_mat is False:
            if int(torch.__version__.split(".")[1]) >= 4:
                module.register_buffer(
                    "%s_cov_mat_sqrt" % name,
                    torch.zeros(max_num_models, data.numel()).cuda(),
                )
            else:
                module.register_buffer(
                    "%s_cov_mat_sqrt" % name,
                    torch.autograd.Variable(
                        torch.zeros(max_num_models, data.numel()).cuda()
                    ),
                )

        params.append((module, name))


class Laplace(torch.nn.Module):
    def __init__(self, base, max_num_models=20, no_cov_mat=False, *args, **kwargs):
        super(Laplace, self).__init__()
        self.params = list()

        self.base = base(*args, **kwargs)
        self.max_num_models = max_num_models

        self.base.apply(
            lambda module: laplace_parameters(
                module=module,
                params=self.params,
                no_cov_mat=no_cov_mat,
                max_num_models=max_num_models,
            )
        )

    def forward(self, input):
        return self.base(input)

    def sample(self, scale=1.0, cov=False, require_grad=False):
        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)
            var = module.__getattr__("%s_var" % name)

            if not cov:
                eps = mean.new(mean.size()).normal_()
                w = mean + scale * torch.sqrt(var) * eps
            else:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                eps = (
                    torch.zeros(cov_mat_sqrt.size(0), 1).normal_().cuda()
                )  # rank-deficient normal results
                # sqrt(max_num_models) scaling comes from covariance matrix
                w = mean + (
                    scale / ((self.max_num_models - 1) ** 0.5)
                ) * var * cov_mat_sqrt.t().matmul(eps).view_as(mean)

            if require_grad:
                w.requires_grad_()
            module.__setattr__(name, w)
            getattr(module, name)
        else:
            for module, name in self.params:
                mean = module.__getattr__("%s_mean" % name)
                var = module.__getattr__("%s_var" % name)

    def export_numpy_params(self):
        mean_list = []
        var_list = []
        for module, name in self.params:
            mean_list.append(module.__getattr__("%s_mean" % name).cpu().numpy().ravel())
            var_list.append(module.__getattr__("%s_var" % name).cpu().numpy().ravel())
        mean = np.concatenate(mean_list)
        var = np.concatenate(var_list)
        return mean, var

    def import_numpy_mean(self, w):
        k = 0
        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)
            s = np.prod(mean.shape)
            mean.copy_(mean.new_tensor(w[k : k + s].reshape(mean.shape)))
            k += s

    def import_numpy_cov_mat_sqrt(self, w):
        k = 0
        for (module, name), sq in zip(self.params, w):
            cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
            cov_mat_sqrt.copy_(cov_mat_sqrt.new_tensor(sq.reshape(cov_mat_sqrt.shape)))

    def estimate_variance(self, loader, criterion, samples=1, tau=5e-4):
        fisher_diag = dict()
        for module, name in self.params:
            var = module.__getattr__("%s_var" % name)
            fisher_diag[(module, name)] = var.new(var.size()).zero_()
        self.sample(scale=0.0, require_grad=True)
        for s in range(samples):
            t_s = time.time()
            for input, target in loader:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                output = self(input)
                distribution = torch.distributions.Categorical(logits=output)
                y = distribution.sample()
                loss = criterion(output, y)

                loss.backward()

                for module, name in self.params:
                    grad = module.__getattr__(name).grad
                    fisher_diag[(module, name)].add_(torch.pow(grad, 2))
            t = time.time() - t_s
            print("%d/%d %.2f sec" % (s + 1, samples, t))

        for module, name in self.params:
            f = fisher_diag[(module, name)] / samples
            var = 1.0 / (f + tau)
            module.__getattr__("%s_var" % name).copy_(var)

    def scale_grid_search(
        self, loader, criterion, logscale_range=torch.arange(-10, 0, 0.5).cuda()
    ):
        all_losses = torch.zeros_like(logscale_range)
        t_s = time.time()
        for i, logscale in enumerate(logscale_range):
            print("forwards pass with ", logscale)
            current_scale = torch.exp(logscale)
            self.sample(scale=current_scale)

            result = eval(loader, self, criterion)

            all_losses[i] = result["loss"]

        min_index = torch.min(all_losses, dim=0)[1]
        scale = torch.exp(logscale_range[min_index]).item()
        t_s_final = time.time() - t_s
        print("estimating scale took %.2f sec" % (t_s_final))
        return scale
