import torch
import numpy as np


def to_sparse(x):
    """ converts dense tensor x to sparse format  
        from https://discuss.pytorch.org/t/how-to-convert-a-dense-matrix-to-a-sparse-one/7809/2 """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if sum(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)

    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    print(indices, values)
    return sparse_tensortype(indices, values, x.size())


def swag_parameters(module, params, no_cov_mat=True):
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            print(module, name)
            continue
        data = module._parameters[name].data
        module._parameters.pop(name)
        module.register_buffer('%s_mean' % name, data.new(data.size()).zero_())
        module.register_buffer('%s_sq_mean' % name, data.new(data.size()).zero_())
        if no_cov_mat is False:
            module.register_buffer('%s_cov_mat_sqrt' % name, torch.zeros(0, data.numel()))

        params.append((module, name))


class SWAG(torch.nn.Module):
    def __init__(self, base, no_cov_mat = True, *args, **kwargs):
        super(SWAG, self).__init__()

        self.register_buffer('n_models', torch.zeros([1]))
        self.params = list()

        self.no_cov_mat = no_cov_mat

        self.base = base(*args, **kwargs)
        self.base.apply(lambda module: swag_parameters(module, self.params, self.no_cov_mat))

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
        #print(self.n_models.size())
        for (module, name), base_param in zip(self.params, base_model.parameters()):
            mean = module.__getattr__('%s_mean' % name)
            sq_mean = module.__getattr__('%s_sq_mean' % name)
            

            #first moment
            mean = mean * self.n_models / (self.n_models + 1.0) + base_param.data / (self.n_models + 1.0)
            
            #second moment
            sq_mean = sq_mean * self.n_models / (self.n_models + 1.0) + base_param.data ** 2 / (self.n_models + 1.0)

            if self.no_cov_mat is False:
                cov_mat_sqrt = module.__getattr__('%s_cov_mat_sqrt' % name)
                
                #block covariance matrices, naive way of doing this
                dev = (base_param.data - mean).view(-1,1)
                
                cov_mat_sqrt._values().mul_(self.n_models / (self.n_models + 1.0))
                #cov_mat +=  to_sparse( dev.mul(dev.t()) * self.n_models / ((self.n_models + 1.0) ** 2) ).cuda()
                #reference for this update: http://www.cs.columbia.edu/~djhsu/papers/gauss.pdf
                #tallying this at the end
                cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1,1)),dim=-1)
                module.__setattr__('%s_cov_mat_sqrt' % name, cov_mat_sqrt)

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