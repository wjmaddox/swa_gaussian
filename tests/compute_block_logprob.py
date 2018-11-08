import torch 
import gpytorch

import sys
sys.path.append('..')
import data
import utils
import models
import swag

import copy

from gpytorch.lazy import RootLazyTensor, DiagLazyTensor, AddedDiagLazyTensor
from gpytorch.distributions import MultivariateNormal

def flatten(lst):
    tmp = [i.contiguous().view(-1,1) for i in lst]
    return torch.cat(tmp).view(-1)

torch.backends.cudnn.benchmark = False

model_cfg = getattr(models, 'VGG16')
num_classes = 10
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model = model.cuda()

swag_model = swag.SWAG(model_cfg.base, no_cov_mat=False, max_num_models=20, loading = True, 
                       *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
swag_model = swag_model.cuda()

checkpoint = torch.load('/home/wesley/Desktop/nfs01_tesla/swa_uncertainties/exps/vgg16_cifar10_0618_1/swag-300.pt')

swag_model.load_state_dict(checkpoint['state_dict'])

mean_list = []
var_list = []
cov_mat_root_list = []
for module, name in swag_model.params:
    mean = module.__getattr__('%s_mean' % name)
    sq_mean = module.__getattr__('%s_sq_mean' % name)
    cov_mat_sqrt = module.__getattr__('%s_cov_mat_sqrt' % name)
    
    mean_list.append(copy.deepcopy(mean))
    var_list.append(copy.deepcopy(sq_mean - mean ** 2.0))
    cov_mat_root_list.append(copy.deepcopy(cov_mat_sqrt))

#now sample a random variable
swag_model.sample(scale=1.0, cov=True, seed=1107)
param_list = [getattr(param, name) for param, name in swag_model.params]

def compute_ll_for_block(vec, mean, var, cov_mat_root):
    vec = flatten(vec).contiguous().cpu()
    mean = flatten(mean).contiguous().cpu()
    var = flatten(var).contiguous().cpu()
    cov_mat_root = cov_mat_root.contiguous().cpu()
    
    cov_mat_lt = RootLazyTensor(cov_mat_root.t())
    var_lt = DiagLazyTensor(var + 1e-6)
    covar_lt = AddedDiagLazyTensor(var_lt, cov_mat_lt)
    qdist = MultivariateNormal(mean, covar_lt)
    with gpytorch.settings.max_preconditioner_size(20) and gpytorch.settings.max_cg_iterations(100) and gpytorch.settings.max_root_decomposition_size(99):
        return qdist.log_prob(vec)

def block_logll(param_list):
    full_logprob = 0
    for i, (param, mean, var, cov_mat_root) in enumerate(zip(param_list, mean_list, var_list, cov_mat_root_list)):
        print('Block: ', i)
        block_ll = compute_ll_for_block(param, mean, var, cov_mat_root)
        full_logprob += block_ll

    return full_logprob

def full_logll(param_list):
    cov_mat_root = torch.cat(cov_mat_root_list,dim=1).contiguous().cpu()
    mean_vector = flatten(mean_list).contiguous().cpu()
    var_vector = flatten(var_list).contiguous().cpu()
    param_vector =flatten(param_list).contiguous().cpu()
    return compute_ll_for_block(param_vector, mean_vector, var_vector, cov_mat_root)
    

block_logprob = block_logll(param_list)
print('Block logprob: ', block_logprob)

full_logprob = full_logll(param_list)
print('Full logprob: ', full_logprob)