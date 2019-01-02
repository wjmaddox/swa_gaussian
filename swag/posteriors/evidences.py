import torch

from datetime import datetime
import math
from gpytorch.lazy import RootLazyTensor, DiagLazyTensor, AddedDiagLazyTensor
import torch.nn.functional as F

from ..utils import bn_update, flatten

def LogSumExp(x,dim=0):
    m,_ = torch.max(x,dim=dim,keepdim=True)
    return m + torch.log((x - m).exp().sum(dim=dim,keepdim=True))

def compute_numparams(model):
    numparams = 0
    for module, name in model.params:
        numparams += getattr(module, name).numel()
    return numparams

def compute_swag_param_norm(model):
    l2norm = 0
    for module, name in model.params:
        w = getattr(module, name)
        l2norm += flatten(w).norm()
    return l2norm

def compute_epoch_logprobs(loaders, swa_model, block=False, use_test = True, samples = 10, cov=True, scale = 1.0, wd_scale=3e-4):
    r"""loader: dataset loader
    swa_model: stochastic weight averaging model
    criterion: loss function
    samples: number of samples to draw from laplace approximation
    scale: multiple to scale the variance of laplace approximation by
    cov: whether to use the estimated covariance matrix 
    wd_scale: scaling on normal prior"""

    if use_test:
        which_loader = 'test'
    else:
        which_loader = 'train'

    loss_sum = 0.0
    correct = 0.0
    #set seed by time + range(samples) basically
    seed_base = int(datetime.now().timestamp())

    epoch_logprob_list = []
    epoch_logprior_list = []
    epoch_logapprox_list = [] #doing this here for seeds

    swa_model.eval()

    with torch.no_grad():
        for i in range(samples):
            print(i)
            #randomly sample from N(swa, swa_var) with seed i
            swa_model.sample(scale=scale, block=False, cov=cov, seed=i+seed_base)

            #perform batch norm update with training
            bn_update(loaders['train'], swa_model)

            batch_prob_list = []

            for j, (input, target) in enumerate(loaders[which_loader]):
                input = input.cuda(async=True)
                target = target.cuda(async=True)

                #compute 0 mean gaussian prior on the weights during the first batch
                if j is 0:
                    #compute wd_scale * norm(w) = N(0, 1/wd_scale I)
                    l2norm = compute_swag_param_norm(swa_model)
                    prior_loss = l2norm * wd_scale

                    #append to list
                    epoch_logprior_list.append(prior_loss.view(-1,1))

                    #compute q(\theta|{x,y}) with current seed
                    epoch_logapprox_list.append(swa_model.compute_logprob().view(-1,1))

                #standard forwards pass but we don't reduce
                output = swa_model(input)
                loss = F.cross_entropy(output, target, reduction='none')

                #print(loss.size())
                batch_prob_list.append(loss.unsqueeze(1))

            #stack all results
            batch_logprobs = torch.cat(batch_prob_list)
            epoch_logprob_list.append(batch_logprobs)

    epoch_logprob = torch.cat(epoch_logprob_list, dim=1)
    epoch_logprior = torch.cat(epoch_logprior_list)
    epoch_logapprox = torch.cat(epoch_logapprox_list)

    # \sum_k (\sum_i log p(y_i| \theta_k) ) + log p(\theta_k) 
    log_joint_total = epoch_logprob.sum() + epoch_logprior.sum()
    #print(epoch_logprob.size(), epoch_logprior.size(), epoch_logapprox.size())

    return {
        'log_joint': -log_joint_total,
        'log_ll': -epoch_logprob.sum(dim=0).unsqueeze(1),
        'log_prior': -epoch_logprior,
        'log_q': -epoch_logapprox,
        'full_log_ll': -epoch_logprob
    } 

def log_marginal_laplace(log_joint_swa, logdet, model_numparams):
    #compute number of parameters of model
    #model_numparams = compute_num_params(swa_model)

    #log(2pi) = 1.83
    # p/2 * log(2pi)
    normalizing_constant = model_numparams/2.0 * 1.8378770664093453

    # 1/2 * log|\Sigma|
    logdet_term = 0.5 * logdet

    # p/2 * log(2pi) + 1/2 * log|\Sigma| + log(p(Y|X,\bar{\theta})p(\bar{\theta}))
    laplace_estimate = normalizing_constant + logdet_term + log_joint_swa

    return laplace_estimate

def log_marginal_bartlett(log_joint_swa, logdet, model_numparams, log_joint_samples, num_samples):
    #model_numparams = compute_num_params(swa_model)
    
    # -p/2 * log{p}
    correction_constant = model_numparams/2.0 * (- math.log(model_numparams))

    #first compute laplace term
    # laplace
    laplace_term = log_marginal_laplace(log_joint_swa, logdet, model_numparams)

    #joint = 2 * (log{p(Y, \bar{\theta}| X)} - 1/K \sum^K log{p(Y, \bar{\theta}|X)}
    joint_sampled_mean = 2.0 * (log_joint_swa - log_joint_samples/num_samples)

    # laplace - p/2 * log{p} + p/2 * joint
    bartlett_estimate = correction_constant + laplace_term + model_numparams/2.0 * joint_sampled_mean.abs().log()

    return bartlett_estimate
    
def log_marginal_is(log_ll, log_prior, log_approx, num_samples):
    #assert log_ll.size() == log_prior.size()

    log_var_approx = log_ll.view(-1,1) + log_prior.view(-1,1) - log_approx.view(-1,1)
    print(log_var_approx.size())
    is_estimate = LogSumExp(log_var_approx) - math.log(num_samples)

    return -is_estimate

def log_marginal_elbo(log_ll_samples, num_samples, model_numparams, logdet, swa_model, wd_scale = 3e-4):
    #1/K \sum^K log(p(Y|X,\theta_k))
    elbo_likelihood = log_ll_samples.sum()/num_samples

    _, var_list, cov_mat_root_list = swa_model.generate_mean_var_covar()

    #tr(sigma)
    # tr(sigma) = tr(blocked sigma), because sum of diagonal
    trace_comp = 0
    for (var, cov_mat_root) in zip(var_list, cov_mat_root_list):
        cov_mat_lt = RootLazyTensor(cov_mat_root.t())
        var_lt = DiagLazyTensor(var + 1e-6)
        covar_lt = AddedDiagLazyTensor(var_lt, cov_mat_lt)
        trace_comp += covar_lt.diag().sum()

    # \mu^T \mu
    swa_norm = compute_swag_param_norm(swa_model)

    #2KL(q||p) = 1/\sigma (tr(\Sigma) + \mu^T \mu ) - p + p log \sigma - \log|\Sigma|
    kl_elbo = 0.5 * (1/wd_scale * ( trace_comp +  swa_norm ) - model_numparams + \
                            model_numparams * math.log(wd_scale) - logdet)

    print('KL(q||p): ', kl_elbo)
    print('log(p(Y|X, theta))', elbo_likelihood)

    return elbo_likelihood - kl_elbo

def compute_dic1(log_ll_samples, log_ll_swa, num_samples):
    sample_deviance = -2.0/num_samples * log_ll_samples.sum()
    expected_deviance = 2.0 * log_ll_swa

    # -2/K \sum^K log{p(Y|\theta, X)} + 2 log{p(Y|\bar{\theta},X)}
    neff = expected_deviance + sample_deviance

    print('DIC effective parameters: ', neff)
    # neff + -2/K \sum^K log{p(Y|\theta, X)}
    return neff + sample_deviance, neff

def compute_waic(log_ll_full, num_samples):
    print(log_ll_full.size())
    # neff_waic = 2 \sum^N V(log{p(y_i |\theta, x_i)})
    sample_vars = torch.var(log_ll_full, dim=1)
    neff_waic = sample_vars.sum()

    print('WAIC effective parameters: ', neff_waic)

    elppd = LogSumExp(log_ll_full, dim=1).sum() - log_ll_full.size(0) * math.log(num_samples)

    return elppd - neff_waic, neff_waic

