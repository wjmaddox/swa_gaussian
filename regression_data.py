import torch
import torch.distributions as td

def generate_toy_problem(N=20, noise_sd=3.0, train=False, **kwargs):
    #looks like we use the same dataset for training and testing

    #set seed
    if train:
        torch.manual_seed(1)
    else:
        torch.manual_seed(2)

    #generate data according to Hernandez-Lobato et al 2015
    x = td.uniform.Uniform(torch.tensor([-4.0]), torch.tensor([4.0]) ).sample((N,1))
    y = x**3 + torch.distributions.normal.Normal(torch.zeros(1), torch.tensor([noise_sd])).sample((N,1))

    x = x.view(-1,1)
    y = y.view(-1,1)

    #create dataset
    dataset = torch.utils.data.TensorDataset(x, y)
    if train:
        #print('y is:', y)
        setattr(dataset, 'train_data', x)
        setattr(dataset, 'train_labels', y)
        #print(dataset.train_labels)
    else:
        setattr(dataset, 'test_data', x)
        setattr(dataset, 'test_labels', y)

    return dataset