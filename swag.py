import torch
import numpy as np


def swag_parameters(module, params):
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            print(module, name)
            continue
        data = module._parameters[name].data
        module._parameters.pop(name)
        module.register_buffer('%s_mean' % name, data.new(data.size()).zero_())
        module.register_buffer('%s_sq_mean' % name, data.new(data.size()).zero_())
        params.append((module, name))


class SWAG(torch.nn.Module):
    def __init__(self, base, *args, **kwargs):
        super(SWAG, self).__init__()

        self.register_buffer('n_models', torch.zeros([1]))
        self.params = list()

        self.base = base(*args, **kwargs)
        self.base.apply(lambda module: swag_parameters(module, self.params))

    def forward(self, input):
        return self.base(input)

    def sample(self, scale=1.0, block_cov=False):
        if block_cov is False:
            for module, name in self.params:
                mean = module.__getattr__('%s_mean' % name)
                sq_mean = module.__getattr__('%s_sq_mean' % name)
                eps = mean.new(mean.size()).normal_()
                w = mean + scale * eps * torch.sqrt(sq_mean - mean ** 2)
                module.__setattr__(name, w)
        # else:
        #    for module, name in self.params:
        #        mean = module.__getattr__('%s_mean', % name)

    def collect_model(self, base_model):
        for (module, name), base_param in zip(self.params, base_model.parameters()):
            mean = module.__getattr__('%s_mean' % name)
            sq_mean = module.__getattr__('%s_sq_mean' % name)
            mean = mean * self.n_models / (self.n_models + 1.0) + base_param.data / (self.n_models + 1.0)
            sq_mean = sq_mean * self.n_models / (self.n_models + 1.0) + base_param.data ** 2 / (self.n_models + 1.0)
            module.__setattr__('%s_mean' % name, mean)
            module.__setattr__('%s_sq_mean' % name, sq_mean)
        self.n_models.add_(1.0)

    def export_numpy_params(self):
        mean_list = []
        sq_mean_list = []
        for module, name in self.params:
            mean_list.append(module.__getattr__('%s_mean' % name).cpu().numpy().ravel())
            sq_mean_list.append(module.__getattr__('%s_sq_mean' % name).cpu().numpy().ravel())
        mean = np.concatenate(mean_list)
        sq_mean = np.concatenate(sq_mean_list)
        var = sq_mean - np.square(mean)
        return mean, var

    def import_numpy_weights(self, w):
        k = 0
        for module, name in self.params:
            mean = module.__getattr__('%s_mean' % name)
            s = np.prod(mean.shape)
            module.__setattr__(name, mean.new_tensor(w[k:k+s].reshape(mean.shape)))
            k += s




