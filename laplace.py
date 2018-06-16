import torch
import numpy as np
import torch.distributions
import time


def laplace_parameters(module, params):
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            print(module, name)
            continue
        data = module._parameters[name].data
        module._parameters.pop(name)
        module.register_buffer('%s_mean' % name, data.new(data.size()).zero_())
        module.register_buffer('%s_var' % name, data.new(data.size()).zero_())
        module.register_buffer(name, data.new(data.size()).zero_())

        params.append((module, name))


class Laplace(torch.nn.Module):
    def __init__(self, base, *args, **kwargs):
        super(Laplace, self).__init__()

        self.params = list()

        self.base = base(*args, **kwargs)
        self.base.apply(lambda module: laplace_parameters(module=module, params=self.params))

    def forward(self, input):
        return self.base(input)

    def sample(self, scale=1.0, require_grad=False):
        for module, name in self.params:
            mean = module.__getattr__('%s_mean' % name)
            var = module.__getattr__('%s_var' % name)
            eps = mean.new(mean.size()).normal_()
            w = mean + scale * torch.sqrt(var) * eps
            if require_grad:
                w.requires_grad_()
            module.__setattr__(name, w)
            getattr(module, name)

    def export_numpy_params(self):
        mean_list = []
        var_list = []
        for module, name in self.params:
            mean_list.append(module.__getattr__('%s_mean' % name).cpu().numpy().ravel())
            var_list.append(module.__getattr__('%s_var' % name).cpu().numpy().ravel())
        mean = np.concatenate(mean_list)
        var = np.concatenate(var_list)
        return mean, var

    def import_numpy_mean(self, w):
        k = 0
        for module, name in self.params:
            mean = module.__getattr__('%s_mean' % name)
            s = np.prod(mean.shape)
            mean.copy_(mean.new_tensor(w[k:k + s].reshape(mean.shape)))
            k += s

    def estimate_variance(self, loader, criterion, samples=1, tau=5e-4):
        fisher_diag = dict()
        for module, name in self.params:
            var = module.__getattr__('%s_var' % name)
            fisher_diag[(module, name)] = var.new(var.size()).zero_()
        self.sample(scale=0.0, require_grad=True)
        for s in range(samples):
            t_s = time.time()
            for input, target in loader:
                input = input.cuda(async=True)
                target = target.cuda(async=True)

                output = self(input)
                distribution = torch.distributions.Categorical(logits=output)
                y = distribution.sample()
                loss = criterion(output, y)

                loss.backward()

                for module, name in self.params:
                    grad = module.__getattr__(name).grad
                    fisher_diag[(module, name)].add_(torch.pow(grad, 2))
            t = time.time() - t_s
            print('%d/%d %.2f sec' % (s + 1, samples, t))

        for module, name in self.params:
            f = fisher_diag[(module, name)] / samples
            var = 1.0 / (f  + tau)
            module.__getattr__('%s_var' % name).copy_(var)





