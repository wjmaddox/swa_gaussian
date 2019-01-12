import torch
import numpy as np
import itertools
from torch.distributions.normal import Normal
import copy

import gpytorch
from gpytorch.lazy import RootLazyTensor, DiagLazyTensor, AddedDiagLazyTensor
from gpytorch.distributions import MultivariateNormal

from ..utils import flatten

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
            module.register_buffer('%s_cov_mat_sqrt' % name, torch.zeros(num_models,data.numel()).cuda())

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

    def sample(self, scale=1.0, cov=False, seed=None, block = False, fullrank = True):
        if seed is not None:
            torch.manual_seed(seed)

        if not block:
            self.sample_fullrank(scale, cov, fullrank)
        else:
            self.sample_blockwise(scale, cov, fullrank)
    
    def sample_blockwise(self, scale, cov, fullrank):
        for module, name in self.params:
            mean = module.__getattr__('%s_mean' % name)

            sq_mean = module.__getattr__('%s_sq_mean' % name)
            eps = mean.new(mean.size()).normal_()
            diag_sample = scale * torch.sqrt(torch.clamp(sq_mean - mean ** 2, min=0.)) * eps
            #print( (sq_mean.double() - (mean.double() ** 2)).min() )

            if cov is True:
                cov_mat_sqrt = module.__getattr__('%s_cov_mat_sqrt' % name)
                eps = torch.zeros(cov_mat_sqrt.size(0), 1).normal_().cuda() #rank-deficient normal results
                cov_sample = (scale/((self.max_num_models - 1) ** 0.5)) * cov_mat_sqrt.t().matmul(eps).view_as(mean)

                if fullrank:
                    w = mean + diag_sample + cov_sample
                else:
                    w = mean + diag_sample 

            else:                
                w = mean + diag_sample

            module.__setattr__(name, w)

    def sample_fullrank(self, scale, cov, fullrank):
        #different sampling procedure to prevent block-diagonal gaussians from being sampled
        if cov is True and scale != 0.0:
            #combine all cov mats into a list
            cov_mat_sqrt_list = []
            mean_list = []
            for module, name in self.params:
                mean_current = module.__getattr__('%s_mean' % name)
                mean_list.append(mean_current)

                cov_mat_sqrt_current = module.__getattr__('%s_cov_mat_sqrt' % name)
                #cov_mat_sqrt_list.append(cov_mat_sqrt_current)
                cov_mat_sqrt_list.append(cov_mat_sqrt_current.cpu())
            
            #now flatten the covariances into a matrix
            cov_mat_sqrt = torch.cat(cov_mat_sqrt_list,dim=1)
            #eps = torch.zeros(cov_mat_sqrt.size(0), 1).normal_().cuda() #rank-deficient normal results
            eps = torch.zeros(cov_mat_sqrt.size(0), 1).normal_() #rank-deficient normal results
            zero_mean_samples = (scale/((self.max_num_models - 1) ** 0.5)) * cov_mat_sqrt.t().matmul(eps)
            zero_mean_samples = zero_mean_samples.cuda()

            #unflatten the covariances back into a list
            zero_mean_samples_list = unflatten_like(zero_mean_samples.t(), mean_list)
            del mean_list

        if cov is not True:
            iterator = zip(self.params, self.params)
        else:
            iterator = zip(self.params, zero_mean_samples_list)

        for (module, name), sample in iterator:                    
            mean = module.__getattr__('%s_mean' % name)
#             sample = sample.cuda()
            if scale == 0.0:
                w = mean
            else:
                #print('here cov is', cov)
                if cov is True:
                    sq_mean = module.__getattr__('%s_sq_mean' % name)
                    eps = mean.new(mean.size()).normal_()

                    if fullrank:
                        #see Section 3.3 of variational boosting
                        #Cov(D'z_1 + sigma I z_2) = DD' + sigma I
                        w = mean + sample.view_as(mean) + torch.sqrt(sq_mean - mean ** 2) * eps
                    else:
                        w = mean + sample.view_as(mean)

                else:
                    sq_mean = module.__getattr__('%s_sq_mean' % name)
                    eps = mean.new(mean.size()).normal_()
                    w = mean + scale * torch.sqrt(sq_mean - mean ** 2) * eps
            module.__setattr__(name, w)

    def collect_model(self, base_model):
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
                dev = (base_param.data - mean).view(-1,1)
                cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1,1).t()),dim=0)

                #remove first column if we have stored too many models
                if (self.n_models+1) > self.max_num_models:
                    cov_mat_sqrt = cov_mat_sqrt[1:, :]
                module.__setattr__('%s_cov_mat_sqrt' % name, cov_mat_sqrt)

            module.__setattr__('%s_mean' % name, mean)
            module.__setattr__('%s_sq_mean' % name, sq_mean)
        self.n_models.add_(1.0)

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

    def generate_mean_var_covar(self):
        mean_list = []
        var_list = []
        cov_mat_root_list = []
        for module, name in self.params:
            mean = module.__getattr__('%s_mean' % name)
            sq_mean = module.__getattr__('%s_sq_mean' % name)
            cov_mat_sqrt = module.__getattr__('%s_cov_mat_sqrt' % name)
            
            mean_list.append(mean)
            var_list.append(sq_mean - mean ** 2.0)
            cov_mat_root_list.append(cov_mat_sqrt)
        return mean_list, var_list, cov_mat_root_list

    def compute_ll_for_block(self, vec, mean, var, cov_mat_root):
        vec = flatten(vec)
        mean = flatten(mean)
        var = flatten(var)

        cov_mat_lt = RootLazyTensor(cov_mat_root.t())
        var_lt = DiagLazyTensor(var + 1e-6)
        covar_lt = AddedDiagLazyTensor(var_lt, cov_mat_lt)
        qdist = MultivariateNormal(mean, covar_lt)

        with gpytorch.settings.num_trace_samples(1) and gpytorch.settings.max_cg_iterations(25):
            return qdist.log_prob(vec)

    def block_logdet(self, var, cov_mat_root):
        var = flatten(var)

        cov_mat_lt = RootLazyTensor(cov_mat_root.t())
        var_lt = DiagLazyTensor(var + 1e-6)
        covar_lt = AddedDiagLazyTensor(var_lt, cov_mat_lt)

        return covar_lt.log_det()

    def block_logll(self,param_list, mean_list, var_list, cov_mat_root_list):
        full_logprob = 0
        for i, (param, mean, var, cov_mat_root) in enumerate(zip(param_list, mean_list, var_list, cov_mat_root_list)):
            #print('Block: ', i)
            block_ll = self.compute_ll_for_block(param, mean, var, cov_mat_root)
            full_logprob += block_ll

        return full_logprob

    def full_logll(self,param_list, mean_list, var_list, cov_mat_root_list):
        cov_mat_root = torch.cat(cov_mat_root_list,dim=1)
        mean_vector = flatten(mean_list)
        var_vector = flatten(var_list)
        param_vector = flatten(param_list)
        return self.compute_ll_for_block(param_vector, mean_vector, var_vector, cov_mat_root)

    def compute_logdet(self, block=False):
        _, var_list, covar_mat_root_list = self.generate_mean_var_covar()

        if block:
            full_logdet = 0
            for (var, cov_mat_root) in zip(var_list, covar_mat_root_list):
                block_logdet = self.block_logdet(var, cov_mat_root)
                full_logdet += block_logdet
        else:
            var_vector = flatten(var_list)
            cov_mat_root = torch.cat(covar_mat_root_list,dim=1)
            full_logdet = self.block_logdet(var_vector, cov_mat_root)

        return full_logdet

    def diag_logll(self, param_list, mean_list, var_list):
        logprob = 0.0
        for param, mean, scale in zip(param_list, mean_list, var_list):
            logprob += Normal(mean, scale).log_prob(param).sum()
        return logprob

    def compute_logprob(self, vec=None, block=False, diag=False):
        mean_list, var_list, covar_mat_root_list = self.generate_mean_var_covar()

        if vec is None:
            param_list = [getattr(param, name) for param, name in self.params]
        else:
            param_list = unflatten_like(vec, mean_list)
        
        if diag:
            return self.diag_logll(param_list, mean_list, var_list)
        elif block is True:
            return self.block_logll(param_list,mean_list, var_list, covar_mat_root_list)
        else:
            return self.full_logll(param_list,mean_list, var_list, covar_mat_root_list)
