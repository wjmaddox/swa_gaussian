import torch
import numpy as np
import itertools
from torch.distributions.normal import Normal

def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i=0
    for tensor in likeTensorList:
        #n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:,i:i+n].view(tensor.shape))
        i+=n
    return outList

def swag_parameters(module, params, no_cov_mat=True, num_models=0):
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            print(module, name)
            continue
        data = module._parameters[name].data
        module._parameters.pop(name)
        module.register_buffer('%s_mean' % name, data.new(data.size()).zero_())
        module.register_buffer('%s_sq_mean' % name, data.new(data.size()).zero_())

        if no_cov_mat is False:
            if int(torch.__version__.split('.')[1]) >= 4:
                #print('max size of cov_mat_sqrt:', num_models, data.numel())
                module.register_buffer('%s_cov_mat_sqrt' % name, torch.zeros(num_models,data.numel()).cuda())
            else:
                module.register_buffer('%s_cov_mat_sqrt' % name, Variable(torch.zeros(num_models,data.numel()).cuda()))

        params.append((module, name))


class SWAG(torch.nn.Module):
    def __init__(self, base, no_cov_mat = True, max_num_models = 0, loading = False, *args, **kwargs):
        super(SWAG, self).__init__()

        self.register_buffer('n_models', torch.zeros([1]))
        self.params = list()

        self.no_cov_mat = no_cov_mat
        self.max_num_models = max_num_models
        if loading is True:
            num_models = self.max_num_models
        else:
            num_models = 0

        self.base = base(*args, **kwargs)
        self.base.apply(lambda module: swag_parameters(module=module, params=self.params, no_cov_mat=self.no_cov_mat, num_models=num_models))

    def forward(self, input):
        return self.base(input)

    def sample(self, scale=1.0, cov=False, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        #different sampling procedure to prevent block based gaussians from being sampled
        if cov is True and scale != 0.0:
            #combine all cov mats into a list
            cov_mat_sqrt_list = []
            mean_list = []
            for module, name in self.params:
                mean_current = module.__getattr__('%s_mean' % name)
                mean_list.append(mean_current)

                cov_mat_sqrt_current = module.__getattr__('%s_cov_mat_sqrt' % name)
                cov_mat_sqrt_list.append(cov_mat_sqrt_current)
            
            #now flatten the covariances into a matrix
            cov_mat_sqrt = torch.cat(cov_mat_sqrt_list,dim=1)
            eps = torch.zeros(cov_mat_sqrt.size(0), 1).normal_().cuda() #rank-deficient normal results
            zero_mean_samples = (scale/((self.max_num_models - 1) ** 0.5)) * cov_mat_sqrt.t().matmul(eps)

            #unflatten the covariances back into a list
            zero_mean_samples_list = unflatten_like(zero_mean_samples.t(), mean_list)
            del mean_list

        if cov is not True:
            iterator = zip(self.params, self.params)
        else:
            iterator = zip(self.params, zero_mean_samples_list)

        for (module, name), sample in iterator:                    
            mean = module.__getattr__('%s_mean' % name)
            if scale == 0.0:
                w = mean
            else:
                #print('here cov is', cov)
                if cov is True:
                    sq_mean = module.__getattr__('%s_sq_mean' % name)
                    eps = mean.new(mean.size()).normal_()

                    w = mean + sample.view_as(mean) + torch.sqrt(sq_mean - mean ** 2) * eps
                else:
                    sq_mean = module.__getattr__('%s_sq_mean' % name)
                    eps = mean.new(mean.size()).normal_()
                    w = mean + scale * torch.sqrt(sq_mean - mean ** 2) * eps
            module.__setattr__(name, w)

    def collect_model(self, base_model, bm=None):
        #print(self.n_models)
        if bm is None:
            for (module, name), base_param in zip(self.params, base_model.parameters()):
                mean = module.__getattr__('%s_mean' % name)
                sq_mean = module.__getattr__('%s_sq_mean' % name)
                
                #first moment
                mean = mean * self.n_models / (self.n_models + 1.0) + base_param.data / (self.n_models + 1.0)

                #second moment
                sq_mean = sq_mean * self.n_models / (self.n_models + 1.0) + base_param.data ** 2 / (self.n_models + 1.0)

                #square root of covariance matrix
                if self.no_cov_mat is False:
                    cov_mat_sqrt = module.__getattr__('%s_cov_mat_sqrt' % name)
                    
                    #block covariance matrices, store deviation from current mean
                    if bm is not None:
                        dev = (bm_param - mean).view(-1,1)
                    else:
                        dev = (base_param.data - mean).view(-1,1)
                    #print(cov_mat_sqrt.size(), dev.size())
                    cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1,1).t()),dim=0)

                    #print(cov_mat_sqrt.size())
                    #remove first column if we have stored too many models
                    if (self.n_models+1) > self.max_num_models:
                        cov_mat_sqrt = cov_mat_sqrt[1:, :]
                        #print(cov_mat_sqrt.size())
                    module.__setattr__('%s_cov_mat_sqrt' % name, cov_mat_sqrt)

                module.__setattr__('%s_mean' % name, mean)
                module.__setattr__('%s_sq_mean' % name, sq_mean)
            self.n_models.add_(1.0)
        else:
            for (module, name), base_param, bp_value in zip(self.params, base_model.parameters(), bm):
                mean = module.__getattr__('%s_mean' % name)
                sq_mean = module.__getattr__('%s_sq_mean' % name)
                
                #first moment
                mean = mean * self.n_models / (self.n_models + 1.0) + base_param.data / (self.n_models + 1.0)
                if torch.sum(torch.isnan(mean)) > 0:
                    print(mean)
                    print(base_param.data)

                #second moment
                sq_mean = sq_mean * self.n_models / (self.n_models + 1.0) + base_param.data ** 2 / (self.n_models + 1.0)

                #square root of covariance matrix
                #if self.no_cov_mat is False:
                cov_mat_sqrt = module.__getattr__('%s_cov_mat_sqrt' % name)
                
                #block covariance matrices, store deviation from current mean
                #if bm is not None:
                
                #print(torch.sum(base_param.data - bp_value))
                dev = (bp_value - mean).view(-1,1)
                #else:
                #    dev = (base_param.data - mean).view(-1,1)
                #print(cov_mat_sqrt.size(), dev.size())
                cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1,1).t()),dim=0)

                #print(cov_mat_sqrt.size())
                #remove first column if we have stored too many models
                if (self.n_models+1) > self.max_num_models:
                    cov_mat_sqrt = cov_mat_sqrt[1:, :]
                    #print(cov_mat_sqrt.size())
                module.__setattr__('%s_cov_mat_sqrt' % name, cov_mat_sqrt)

                module.__setattr__('%s_mean' % name, mean)
                module.__setattr__('%s_sq_mean' % name, sq_mean)
            self.n_models.add_(1.0)
            del bm

    def export_numpy_params(self, export_cov_mat=False):
        mean_list = []
        sq_mean_list = []
        cov_mat_list = []

        for module, name in self.params:
            mean_list.append(module.__getattr__('%s_mean' % name).cpu().numpy().ravel())
            sq_mean_list.append(module.__getattr__('%s_sq_mean' % name).cpu().numpy().ravel())
            if export_cov_mat:
                cov_mat_list.append(module.__getattr__('%s_cov_mat_sqrt' % name).cpu().numpy().ravel())
        mean = np.concatenate(mean_list)
        sq_mean = np.concatenate(sq_mean_list)
        var = sq_mean - np.square(mean)

        if export_cov_mat:
            #cov_mat = np.concatenate(cov_mat_list)
            return mean, var, cov_mat_list
        else:
            return mean, var

    def import_numpy_weights(self, w):
        k = 0
        for module, name in self.params:
            mean = module.__getattr__('%s_mean' % name)
            s = np.prod(mean.shape)
            module.__setattr__(name, mean.new_tensor(w[k:k+s].reshape(mean.shape)))
            k += s

    def compute_logprob(self, diag=True, use_pars=True):
        #currently only diagonal is implemented
        logprob = 0.0

        for (module, name) in self.params:
            #[print(x) for x in module['params']]
            #print(module.weight)
            data = getattr(module, name)
            if use_pars:
                mean = module.__getattr__('%s_mean' % name)
                sq_mean = module.__getattr__('%s_sq_mean' % name)
                scale = torch.sqrt(sq_mean - mean.pow(2.0))
            else:
                mean = torch.zeros_like(data)
                scale = torch.ones_like(data) * 0.0001
            logprob += Normal(mean, scale).log_prob(data).sum()
        
        return logprob