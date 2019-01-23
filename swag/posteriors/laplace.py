"""
    implementation of KFAC Laplace, see reference
    base class ported from: https://github.com/Thrandis/EKFAC-pytorch/kfac.py
"""

import torch
import torch.nn.functional as F
import copy

# Hessian and Jacobian code from: https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                      
def hessian(y, x):                                                                                    
    return jacobian(jacobian(y, x, create_graph=True), x) 

class KFACLaplace(torch.optim.Optimizer):
    r"""KFAC Laplace: based on Scalable Laplace
    Code is partially copied from https://github.com/Thrandis/EKFAC-pytorch/kfac.py.
    TODO: batch norm implementation
    TODO: use some sort of validation set for scaling data_size parameter
    """
    def __init__(self, net, eps, sua=False, pi=False, update_freq=1,
                 alpha=1.0, constraint_norm=False, data_size = 50000, use_batch_norm = False):
        """ K-FAC Preconditionner for Linear and Conv2d layers.
        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.
        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            pi (bool): Computes pi correction for Tikhonov regularization.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter (if == 1, no r. ave.).
            constraint_norm (bool): Scale the gradients by the squared
                fisher norm.
            use_batch_norm: whether or not batch norm layers should be computed
        """
        self.net = net
        self.state = net.state_dict()
        self.mean_state = copy.deepcopy(self.state)
        self.data_size = data_size
        self.use_batch_norm = use_batch_norm

        self.eps = eps
        self.sua = sua
        self.pi = pi
        self.update_freq = update_freq
        self.alpha = alpha
        self.constraint_norm = constraint_norm
        self.params = []
        self._iteration_counter = 0
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d']:
                mod.register_forward_pre_hook(self._save_input)
                mod.register_backward_hook(self._save_grad_output)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                self.params.append(d)

            elif 'BatchNorm' in mod_class and use_batch_norm:
                mod.register_forward_pre_hook(self._save_input)
                mod.register_backward_hook(self._save_grad_output)

                params = [mod.weight, mod.bias]

                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                self.params.append(d)

        super(KFACLaplace, self).__init__(self.params, {})
        #super(KFACLaplace, self).__init__()

    def cuda(self):
        self.net.cuda()

    def load_state_dict(self, checkpoint, **kwargs):
        self.net.load_state_dict(checkpoint, **kwargs)

        self.mean_state = self.net.state_dict()

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()

    def apply(self, *args, **kwargs):
        self.net.apply(*args, **kwargs)

    def sample(self, scale=1.0, **kwargs):

        for group in self.params:
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]

            if 'BatchNorm' in group['layer_type'] and self.use_batch_norm:

                z = torch.zeros_like(weight).normal_()
                sample = state['w_ic'].matmul(z)

                if bias is not None:

                    z = torch.zeros_like(bias).normal_()
                    bias_sample = state['b_ic'].matmul(z)

            else:
                # now compute inverse covariances
                #self._compute_covs(group, state)
                ixxt, iggt, ixxt_chol, iggt_chol = self._inv_covs(state['xxt'], state['ggt'], num_locations=state['num_locations'])
                state['ixxt'] = ixxt
                state['iggt'] = iggt

                # draw samples from AZB
                # appendix B of ritter et al.
                z = torch.randn(state['ixxt'].size(0), state['iggt'].size(0), device = ixxt.device, dtype = ixxt.dtype)
                #z = torch.randn(state['ixxt'].size(0), state['iggt'].size(0), dtype = ixxt.dtype)
                # matmul a z b
                #print(state['ixxt'].shape, state['iggt'].shape)
                sample = ixxt_chol.matmul(z.matmul(iggt_chol)).t()
                #sample = ixxt_chol.cpu().matmul(z.matmul(iggt_chol.cpu())).t()
                sample *= (scale/self.data_size) #scale/N term for inverse
                #sample = sample.cuda()

                if bias is not None:
                    #print(weight.shape, bias.shape, sample.shape)
                    bias_sample = sample[:, -1].contiguous().view(*bias.shape)
                    sample = sample[:, :-1]
                    #print(weight.shape, bias.shape, sample.shape)

            #print(weight.norm(), sample.norm())
            #finally update parameters with new values as mean is current state dict
            weight.data.add_(sample.view_as(weight))
            if bias is not None:
                bias.data.add_(bias_sample.view_as(bias))

    def step(self, update_stats=True, update_params=True):
        #Performs one step of preconditioning.
        fisher_norm = 0.
        for group in self.param_groups:
            #print(torch.cuda.memory_allocated()/(1024**3))
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]

            #print(group['layer_type'])
            if 'BatchNorm' in group['layer_type'] and self.use_batch_norm:
                # now compute hessian of weights
                diag_comp = 100 * weight.size(0) * self.eps * torch.eye(weight.size(0), device = weight.device, dtype = weight.dtype)
                #print(weight.size())
                weight_hessian = jacobian(weight.grad, weight) + diag_comp
                #print(weight_hessian)

                weight_inv_chol = torch.cholesky(weight_hessian)
                state['w_ic'] = weight_inv_chol

                if bias is not None:
                    diag_comp = 100 * self.eps * torch.eye(bias.size(0), device = bias.device, dtype = bias.dtype)
                    bias_hessian = jacobian(bias.grad, bias) + diag_comp

                    state['b_ic'] = torch.cholesky(bias_hessian)

            if group['layer_type'] in ['Linear', 'Conv2d']:

                # Update convariances and inverses
                if update_stats:
                    if self._iteration_counter % self.update_freq == 0:
                        self._compute_covs(group, state)
                        ixxt, iggt, _, _ = self._inv_covs(state['xxt'], state['ggt'],
                                                    state['num_locations'])
                        state['ixxt'] = ixxt
                        state['iggt'] = iggt
                    else:
                        if self.alpha != 1:
                            self._compute_covs(group, state)
                if update_params:
                    # Preconditionning
                    gw, gb = self._precond(weight, bias, group, state)
                    # Updating gradients
                    if self.constraint_norm:
                        fisher_norm += (weight.grad * gw).sum()
                    weight.grad.data = gw
                    if bias is not None:
                        if self.constraint_norm:
                            fisher_norm += (bias.grad * gb).sum()
                        bias.grad.data = gb
                # Cleaning
                if 'x' in self.state[group['mod']]:
                    del self.state[group['mod']]['x']
                if 'gy' in self.state[group['mod']]:
                    del self.state[group['mod']]['gy']
        # Eventually scale the norm of the gradients
        if update_params and self.constraint_norm:
            f_scale = (1. / fisher_norm) ** 0.5
            for group in self.param_groups:
                for param in group['params']:
                    param.grad.data *= f_scale
        if update_stats:
            self._iteration_counter += 1

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        if mod.training:
            self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if mod.training:
            self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        if group['layer_type'] == 'Conv2d' and self.sua:
            return self._precond_sua(weight, bias, group, state)
        ixxt = state['ixxt']
        iggt = state['iggt']
        g = weight.grad.data
        s = g.shape
        if group['layer_type'] == 'Conv2d':
            g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
        g = torch.mm(torch.mm(iggt, g), ixxt)
        if group['layer_type'] == 'Conv2d':
            g /= state['num_locations']
        if bias is not None:
            gb = g[:, -1].contiguous().view(*bias.shape)
            g = g[:, :-1]
        else:
            gb = None
        g = g.contiguous().view(*s)
        return g, gb

    def _precond_sua(self, weight, bias, group, state):
        """Preconditioning for KFAC SUA."""
        ixxt = state['ixxt']
        iggt = state['iggt']
        g = weight.grad.data
        s = g.shape
        mod = group['mod']
        g = g.permute(1, 0, 2, 3).contiguous()
        if bias is not None:
            gb = bias.grad.view(1, -1, 1, 1).expand(1, -1, s[2], s[3])
            g = torch.cat([g, gb], dim=0)
        g = torch.mm(ixxt, g.contiguous().view(-1, s[0]*s[2]*s[3]))
        g = g.view(-1, s[0], s[2], s[3]).permute(1, 0, 2, 3).contiguous()
        g = torch.mm(iggt, g.view(s[0], -1)).view(s[0], -1, s[2], s[3])
        g /= state['num_locations']
        if bias is not None:
            gb = g[:, -1, s[2]//2, s[3]//2]
            g = g[:, :-1]
        else:
            gb = None
        return g, gb

    def _compute_covs(self, group, state):
        """Computes the covariances."""
        mod = group['mod']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']
        # Computation of xxt
        if group['layer_type'] == 'Conv2d':
            if not self.sua:
                x = F.unfold(x, mod.kernel_size, padding=mod.padding,
                             stride=mod.stride)
            else:
                x = x.view(x.shape[0], x.shape[1], -1)
            x = x.data.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
        if self._iteration_counter == 0:
            state['xxt'] = torch.mm(x, x.t()) / float(x.shape[1])
        else:
            state['xxt'].addmm_(mat1=x, mat2=x.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(x.shape[1]))
        # Computation of ggt
        if group['layer_type'] == 'Conv2d':
            gy = gy.data.permute(1, 0, 2, 3)
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state['num_locations'] = 1
        if self._iteration_counter == 0:
            state['ggt'] = torch.mm(gy, gy.t()) / float(gy.shape[1])
        else:
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(gy.shape[1]))

    def _inv_covs(self, xxt, ggt, num_locations):
        """Inverses the covariances."""
        # Computes pi
        pi = 1.0
        if self.pi:
            tx = torch.trace(xxt) * ggt.shape[0]
            tg = torch.trace(ggt) * xxt.shape[0]
            pi = (tx / tg)
        # Regularizes and inverse
        eps = self.eps / num_locations
        diag_xxt = xxt.new(xxt.shape[0]).fill_((eps * pi) ** 0.5)
        diag_ggt = ggt.new(ggt.shape[0]).fill_((eps / pi) ** 0.5)

        # Compute cholesky
        xxt_chol = (xxt + torch.diag(diag_xxt)).cholesky()
        ggt_chol = (ggt + torch.diag(diag_ggt)).cholesky()

        # invert cholesky
        xxt_ichol = torch.inverse(xxt_chol)
        ggt_ichol = torch.inverse(ggt_chol)

        # invert matrix
        ixxt = xxt_ichol.t().matmul(xxt_ichol)
        iggt = ggt_ichol.t().matmul(ggt_ichol)

        return ixxt, iggt, xxt_ichol, ggt_ichol
