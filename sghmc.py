"""
Author: Wesley Maddox
Date: 11/29/17
Copied off of sgd based code
source of sgd: https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py

sghmc as an optimizer, code built off of sgd source code
only update should be the noise
"""

import torch
from torch.optim.optimizer import Optimizer, required
import math

class SGHMC(Optimizer):
    r"""
    Implements stochastic gradient HMC, multiple L values are supported by entering loss as a closure
    
    Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        betahat (float, optional): enables variance updates for gradient (default: 0)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    .. note::
        T?his should correspond to the version of SGHMC 
    __ https://arxiv.org/pdf/1402.4102.pdf that is parameterized with 1-:math:`\rho`
        as the noise of the gradient.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, L = 1, betahat=0.0):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, L = L, betahat=betahat)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGHMC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGHMC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if closure is None and group['L']>1:
                raise ValueError("L > 1 requires a closure")
            
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            betahat = group['betahat']
            
            for l in range(group['L']):
                if closure is not None:
                    loss = closure()
                    
                for p in group['params']:
                    if p.grad is None:
                        continue
                    noise_var = 2.0 * (1.0 - momentum - betahat)
                    d_p = p.grad.data + torch.zeros_like(p.grad.data).normal_().mul(noise_var ** 0.5)
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            #print('here and l is', l)
                            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data).normal_().mul(group['lr'])
                            buf.mul_(momentum).add_(d_p)
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf
                            
                    p.data.add_(-group['lr'], d_p)
        return loss
            #below may or may not be correct
            #want to add in the adam-like varainace update
        """for group in self.param_groups:
            if closure is None and group['L']>1:
                raise ValueError("L > 1 requires a closure")
            
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            #betahat = group['betahat']
            
            beta1, beta2 = 0.9, 0.999
            for _ in range(group['L']):
                if closure is not None:
                    loss = closure()
                    
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    
                    state = self.state[p]
                    #state initialization
                    if len(state)==0:
                        state['step'] = 0
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                        state['exp_avg'] = torch.zeros_like(p.data)
                    
                    state['step']+=1
                    #gradient estimate
                    d_p = p.grad.data
                    
                    state['exp_avg'].mul_(beta1).add_(1 - beta1, d_p)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(1 - beta2, d_p, d_p)
                    
                    
                    #betahat = betahat_sq.pow(0.5) #take the square root
                    mean_sq_exp_avg = (state['exp_avg']/(1.0 - (beta1 ** state['step']))) ** 2
                    var_exp_avg = state['exp_avg_sq']/(1.0- (beta2 ** state['step']))
                    vhat = var_exp_avg - mean_sq_exp_avg
                                
                    betahat = group['lr'] * 0.5 * vhat
                    
                    #print((1-momentum-betahat).min(), (1-momentum-betahat).max())
                    var_term = 2.0 * group['lr'] * (1.0 - momentum - betahat)
                    
                    d_p += torch.zeros_like(p.grad.data).normal_().mul(var_term.pow(0.5))
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                            buf.mul_(momentum).add_(d_p)
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf
                            
                    p.data.add_(-group['lr'], d_p)"""

        

