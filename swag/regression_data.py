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

def generate_boston(train=True, **kwargs):
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_boston

    boston_numpy = load_boston()
    boston_train, boston_test, boston_train_y, boston_test_y = train_test_split(boston_numpy['data'], 
                                                                                boston_numpy['target'], 
                                                                                test_size = 0.1, random_state = 2018)

    boston_train, boston_test  = torch.from_numpy(boston_train).float(), torch.from_numpy(boston_test).float()
    boston_train_y, boston_test_y = torch.from_numpy(boston_train_y).float(), torch.from_numpy(boston_test_y).float()

    boston_train_y, boston_test_y = boston_train_y.view(-1,1), boston_test_y.view(-1,1)
    
    if train:
        dataset = torch.utils.data.TensorDataset(boston_train, boston_train_y)
        setattr(dataset, 'train_data', boston_train)
        setattr(dataset, 'train_labels', boston_train_y)
    else:
        dataset = torch.utils.data.TensorDataset(boston_test, boston_test_y)
        setattr(dataset, 'test_data', boston_test)
        setattr(dataset, 'test_labels', boston_test_y)
    
    return dataset