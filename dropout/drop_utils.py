import torch
from datetime import datetime
from utils import bn_update, LogSumExp

def eval_dropout(loaders, model, criterion, samples = 10):
    r"""loader: dataset loader
    model: stochastic weight averaging model
    criterion: loss function
    samples: number of samples to draw from laplace approximation"""

    loss_sum = 0.0
    correct = 0.0
    model.eval()

    #function that sets dropout to train mode
    def train_dropout(m):
        if type(m)==torch.nn.modules.dropout.Dropout:
            m.train()
    model.apply(train_dropout)

    #set seed by time + range(samples)
    #this is to ensure that we get different random numbers each time
    seed_base = int(datetime.now().timestamp())
    target_list = []
    with torch.no_grad():
        epoch_logprob_list = []
        for i in range(samples):
            torch.manual_seed(i + seed_base)

            batch_prob_list = []

            for j, (input, target) in enumerate(loaders['test']):
                #set seed so dropout should be deterministic by seed
                torch.manual_seed(i + seed_base)

                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                if i==0:
                    target_list.append(target)

                #standard forwards pass but we don't reduce
                output = model(input)
                #loss = criterion(output, target, reduction='none')
                prob = torch.nn.Softmax(dim=1)(output).unsqueeze(1)
                batch_prob_list.append(prob)

            #stack all results
            batch_logprobs = torch.cat(batch_prob_list)
            epoch_logprob_list.append(batch_logprobs)

    epoch_prob = torch.cat(epoch_logprob_list, dim=1).permute(0, 2, 1)

    epoch_logprobs = (epoch_prob.mean(dim=2)+1e-6).log()

    target_stack = torch.cat(target_list)
    pred = epoch_logprobs.data.argmax(1, keepdim=True)
    correct = pred.eq(target_stack.data.view_as(pred)).sum().item()

    loss = criterion(epoch_logprobs, target_stack)

    return {
        'loss': loss.data.item(),
        'accuracy': (correct / len(loaders['test'].dataset)) * 100.0,
    } 