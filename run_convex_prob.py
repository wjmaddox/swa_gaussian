import torch, argparse
import gpytorch
import os, sys
import models
import pdb, math
import torch.nn.functional as F
import torch.nn as nn

from hvp import HVP
from regression_data import generate_cvx_regression
import swag

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--lr_init', type=float, default=0.01, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--dimension', type=int, default=100, help='dimensionality of problem')
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--data', type=int, default=1e5, help='number of datapoints')
parser.add_argument('--swa_start', type=float, default=5, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--scale', type=float, default=10.0, help='scale of l1 regularizaiton')
args = parser.parse_args()

args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

#initialize both model and swag model
class lasso(nn.Module):
    def __init__(self, d=args.dimension):
        super(lasso,self).__init__()
        self.m = nn.Sequential(nn.Linear(d, 1, bias=False))
    def forward(self,x):
        return self.m(x)
print(args.dimension)
model = lasso().to(args.device)
swag_model = swag.SWAG(lasso, no_cov_mat=False, max_num_models=args.epochs - args.swa_start).to(args.device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=0)

dataset, beta, sigmainv = generate_cvx_regression(N=args.data, d=args.dimension)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

for i in range(1, args.epochs + 1):
    for j, (x, y) in enumerate(dataloader):
        x, y = x.to(args.device), y.to(args.device)

        def closure():
            optimizer.zero_grad()

            #compute the l1 norm (ie lasso)
            param_norm = 0.0
            for param in model.parameters():
                param_norm += param.abs().sum()
            
            loss = F.mse_loss(model(x), y)

            #total loss: ||y - beta x||_2^2 - ||beta||_1
            total_loss = loss + args.scale * param_norm/args.data
            total_loss.backward()
            return total_loss
        
        loss = optimizer.step(closure)
    
    #print loss at end of every 10th epoch
    if i%20 is 0:
        print('Epoch: ', i, ' Loss: ', loss.item())

    #save model if after swa start
    if i > (args.swa_start + 1):
        swag_model.collect_model(model)
        
params = list(model.parameters())[0]
print('Last model # of Non-zero Parameters: ' + str(torch.sum(params.abs() > 0.1).cpu().item()))
print('True # of Non-zero Parameters: ' + str(torch.sum(beta.abs() > 0.1).item()))

###now comes the fun part

#true diagonal confidence intervals
def compute_conf_ints(beta, sigmainv, alpha = torch.tensor([0.0125])):
    sigmainv_diag = torch.diag(sigmainv)
    zval = torch.distributions.Normal(torch.zeros(1), torch.ones(1)).icdf(1.0 - alpha/2.0)

    diagonal_ind_conf_ints = (beta - zval * sigmainv_diag/math.sqrt(args.data), beta + zval * sigmainv_diag/math.sqrt(args.data))
    return diagonal_ind_conf_ints

CIs = compute_conf_ints(beta, sigmainv)

#set swag to means
swag_model.sample(0.0)
swag_params = list(swag_model.params[0].parameters())
print(swag_params[0])