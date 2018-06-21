import torch.nn as nn
import torchvision.transforms as transforms
import torch
__all__=['MLP', 'MLPBoston']


class MLPBase(nn.Module):
    def __init__(self, num_classes=0, in_dim=1, layers=2, hidden=7):
        super(MLPBase, self).__init__()

        out_layer_list = [hidden for i in range(layers)]
        if num_classes == 0:
            out_layer_list.append(1) #for regression
        else:
            out_layer_list.append(num_classes)
        
        in_layer_list = [hidden for i in range(layers)]
        in_layer_list.insert(0, in_dim)

        layers = []
        for input, output in zip(in_layer_list, out_layer_list):
            layers.append(nn.Linear(input, output))
            #add relu activations
            layers.append(nn.ReLU())
        layers.pop() #remove final relu layer

        self.model = nn.Sequential(*layers)
        #self.log_noise = nn.Parameter(torch.log(torch.ones(1)*7))

        print(self.model)
        
    def forward(self, x):
        return self.model(x)

class MLP:
    base = MLPBase
    args = list()
    kwargs = {}
    transform_train = transforms.ToTensor()
    transform_test = transforms.ToTensor()

class MLPBoston:
    base = MLPBase
    base.log_noise = nn.Parameter(torch.log(torch.ones(1)*7))
    args = list()
    kwargs = {'in_dim': 13, 'layers': 1, 'hidden':50}
    transform_train = transforms.ToTensor()
    transform_test = transforms.ToTensor()
