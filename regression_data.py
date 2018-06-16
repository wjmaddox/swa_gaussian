import torch
import torch.distributions as td

def generate_toy_problem(N=20, noise_sd=3.0, seed=1, **kwargs):
    #looks like we use the same dataset for training and testing

    #set seed
    torch.manual_seed(seed)

    #generate data according to Hernandez-Lobato et al 2015
    x = td.uniform.Uniform(torch.tensor([-1.0]), torch.tensor([1.0]) ).sample((N,))
    y = x**3 + torch.distributions.normal.Normal(torch.zeros(1), torch.tensor([noise_sd])).sample((N,))

    x = x.view(-1,1)
    y = y.view(-1,1)
    
    #create dataset
    dataset = torch.utils.data.TensorDataset(x, y)
    setattr(dataset, 'train_data', x)
    setattr(dataset, 'test_data', x)
    setattr(dataset, 'train_labels', y)
    setattr(dataset, 'test_labels', y)
    return dataset