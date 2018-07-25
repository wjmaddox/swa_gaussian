import torch
from torch.distributions.normal import Normal
from torch.optim import SGD
import matplotlib.pyplot as plt
import math

torch.manual_seed(2)

def LogSumExp(x,dim=0):
    m,_ = torch.max(x,dim=dim,keepdim=True)
    return m + torch.log((x - m).exp().sum(dim=dim,keepdim=True))

def pathological_mixture(x):
    #x = x * (1.0 + 0.0175 * torch.randn(1))
    mix1 = Normal(torch.zeros(1), torch.tensor([0.5]))
    mix2 = Normal(torch.tensor([1.0]), torch.tensor([0.15]))

    #logsumexp trick
    m1 = mix1.log_prob(x)
    m2 = mix2.log_prob(x).mul(200)

    out = LogSumExp(torch.cat((m1.view(-1,1),m2.view(-1,1)),dim=1),dim=1)
    return out

test_pts = torch.arange(-10.0, 10.0, 0.01)
path_nll = pathological_mixture(test_pts)

def compute_sgd_approx_lr(lr=0.01):
    #initialize
    x = 3.0 * torch.randn(1, requires_grad=False)
    x.requires_grad = True

    ### run sgd
    optim = SGD([x], lr = lr)

    num_steps = 100
    all_x = torch.zeros(num_steps)
    for i in range(num_steps):
        all_x[i] = x.data
        #print(x)
        optim.zero_grad()
        loss = -pathological_mixture(x)
        loss.backward()
        optim.step()

    ## compute swa distribution
    swa_estimate = all_x[int(num_steps/2):].mean()
    swa_std = all_x[int(num_steps/2):].std() * 1/math.sqrt(int(num_steps/2))
    swa_dist = Normal(swa_estimate, swa_std)

    swa_nll = -swa_dist.log_prob(test_pts)

    return swa_nll, swa_estimate, swa_std

swa_nll = None
lr_used = None
loss = 1e10
swa_mean, swa_std = None, None
for lr in range(-40, 0):
    lr = 10 ** (lr/10)
    #print(lr)
    loss_curr = 0.0
    for _ in range(10):
        out, mu, std = compute_sgd_approx_lr(lr=lr)
        loss_curr += (out - path_nll).pow(2).mean()/10

    if loss_curr < loss:
        print(loss, loss_curr)
        loss = loss_curr
        lr_used = lr
        swa_nll = out
        swa_mean, swa_std = mu, std

def compute_second_deriv(x):
    out = pathological_mixture(x)
    grad_val = torch.autograd.grad(out, [x], create_graph=True)
    second_deriv = torch.autograd.grad(grad_val, [x], grad_outputs=torch.ones(1))
    return second_deriv[0]

print('best learning rate: ', lr_used)

#####laplace approximation here
_, laplace_ind = torch.max(path_nll.view(-1,1),0)
laplace_mean = test_pts[laplace_ind]
laplace_mean.requires_grad = True

laplace_std = (-1/compute_second_deriv(laplace_mean)).sqrt()
laplace_dist = Normal(laplace_mean, laplace_std)
laplace_nll = -laplace_dist.log_prob(test_pts).detach().numpy()

### swag-hessian approximation
swa_mean.requires_grad = True
sl_std = -swa_std/compute_second_deriv(swa_mean)
swag_dist = Normal(swa_mean, sl_std)
swag_nll = -swag_dist.log_prob(test_pts).detach().numpy()

##### now plot everything
plt.plot(test_pts.numpy(), path_nll.numpy())
plt.plot(test_pts.numpy(), -swa_nll.numpy(), color='red', linestyle='--')
plt.plot(test_pts.numpy(), -swag_nll, color='orange', linestyle='--')
plt.plot(test_pts.numpy(), -laplace_nll, color='black', linestyle='--')
plt.ylim((-30, 20))
plt.xlim((-5,5))
plt.xlabel('x', fontsize=16)
plt.ylabel('Log-Likelihood', fontsize=16)
plt.savefig('plots/toy_dist.eps')
plt.show()
