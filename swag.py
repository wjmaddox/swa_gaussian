import torch
import numpy as np
import itertools

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
            
        for module, name in self.params:
            mean = module.__getattr__('%s_mean' % name)
            if cov is True:
                cov_mat_sqrt = module.__getattr__('%s_cov_mat_sqrt' % name)
                eps = torch.zeros(cov_mat_sqrt.size(0), 1).normal_().cuda() #rank-deficient normal results
                w = mean + (scale/((self.max_num_models - 1) ** 0.5)) * cov_mat_sqrt.t().matmul(eps).view_as(mean)
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

    # def _load_from_different_state_dict(self, state_dict, prefix, strict, missing_keys, unexpected_keys, error_msgs):
    #     r"""Copies parameters and buffers from :attr:`state_dict` into only
    #     this module, but not its descendants. This is called on every submodule
    #     in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
    #     module in input :attr:`state_dict` is at ``state_dict._metadata[prefix]``.
    #     Subclasses can achieve class-specific backward compatible loading using
    #     the version number at ``state_dict._metadata[prefix]["version"]``.

    #     .. note::
    #         :attr:`state_dict` is not the same object as the input
    #         :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
    #         it can be modified.

    #     Arguments:
    #         state_dict (dict): a dict containing parameters and
    #             persistent buffers.
    #         prefix (str): the prefix for parameters and buffers used in this
    #             module
    #         strict (bool): whether to strictly enforce that the keys in
    #             :attr:`state_dict` with :attr:`prefix` match the names of
    #             parameters and buffers in this module
    #         missing_keys (list of str): if ``strict=False``, add missing keys to
    #             this list
    #         unexpected_keys (list of str): if ``strict=False``, add unexpected
    #             keys to this list
    #         error_msgs (list of str): error messages should be added to this
    #             list, and will be reported together in
    #             :meth:`~torch.nn.Module.load_state_dict`
    #     """
    #     local_name_params = itertools.chain(self._parameters.items(), self._buffers.items())
    #     local_state = {k: v.data for k, v in local_name_params if v is not None}

    #     for name, param in local_state.items():
    #         key = prefix + name
    #         if key in state_dict:
    #             input_param = state_dict[key]
    #             if isinstance(input_param, torch.nn.Parameter):
    #                 # backwards compatibility for serialized parameters
    #                 input_param = input_param.data
    #             try:
    #                 param.resize_as_(input_param).copy_(input_param)
    #             except Exception:
    #                 error_msgs.append('While copying the parameter named "{}", '
    #                                   'whose dimensions in the model are {} and '
    #                                   'whose dimensions in the checkpoint are {}.'
    #                                   .format(key, param.size(), input_param.size()))
    #         elif strict:
    #             missing_keys.append(key)

    #     if strict:
    #         for key, input_param in state_dict.items():
    #             if key.startswith(prefix):
    #                 input_name = key[len(prefix):]
    #                 input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
    #                 if input_name not in self._modules and input_name not in local_state:
    #                     unexpected_keys.append(key)

    # def load_different_state_dict(self, state_dict, strict=True):
    #     r"""Copies parameters and buffers from :attr:`state_dict` into
    #     this module and its descendants. If :attr:`strict` is ``True``, then
    #     the keys of :attr:`state_dict` must exactly match the keys returned
    #     by this module's :meth:`~torch.nn.Module.state_dict` function.

    #     Arguments:
    #         state_dict (dict): a dict containing parameters and
    #             persistent buffers.
    #         strict (bool, optional): whether to strictly enforce that the keys
    #             in :attr:`state_dict` match the keys returned by this module's
    #             :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    #     """
    #     missing_keys = []
    #     unexpected_keys = []
    #     error_msgs = []

    #     # copy state_dict so _load_from_state_dict can modify it
    #     metadata = getattr(state_dict, '_metadata', None)
    #     state_dict = state_dict.copy()
    #     if metadata is not None:
    #         state_dict._metadata = metadata

    #     def load(module, prefix=''):
    #         module._load_from_different_state_dict(
    #             state_dict, prefix, strict, missing_keys, unexpected_keys, error_msgs)
    #         for name, child in module._modules.items():
    #             if child is not None:
    #                 load(child, prefix + name + '.')

    #     load(self)

    #     if strict:
    #         error_msg = ''
    #         if len(unexpected_keys) > 0:
    #             error_msgs.insert(
    #                 0, 'Unexpected key(s) in state_dict: {}. '.format(
    #                     ', '.join('"{}"'.format(k) for k in unexpected_keys)))
    #         if len(missing_keys) > 0:
    #             error_msgs.insert(
    #                 0, 'Missing key(s) in state_dict: {}. '.format(
    #                     ', '.join('"{}"'.format(k) for k in missing_keys)))

    #     if len(error_msgs) > 0:
    #         raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
    #                            self.__class__.__name__, "\n\t".join(error_msgs)))