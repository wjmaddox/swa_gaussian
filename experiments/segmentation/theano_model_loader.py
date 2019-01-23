"""
    theano model loading script
    loading the pre-trained model provided in the 100-layers tiramisu code doesn't give reasonable accuracies
    so there's probably a model definition issue
"""
import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from datasets import camvid
from datasets import joint_transforms

from models.tiramisu import FCDenseNet103
from utils.training import test, train

from swag.utils import bn_update

parser = argparse.ArgumentParser(description='Testing Theano Model')
parser.add_argument('--dataset_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--theano_model_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to theano model location (default: None)')
args = parser.parse_args()

def add_names(lst, fullname):
    # helper function that takes a list of the type (name, param) and 
    # returns list of type (fullname.name, param)

    newlst = []
    for name, param in lst:
        new_name = fullname+'.'+name
        newlst.append( (new_name, param) )
    return newlst

def load_theano_fcmodel(model, theano_dict):
    # first load parameter values in numeric order
    with theano_dict as f:
        saved_params_values = [f['arr_%d' % i] for i in range(len(f.files))]

    # now remove bad parameters
    no_bad_params = []
    for param in saved_params_values:
        if param.sum()==0 or param.mean()==1:
            continue
        else:
            no_bad_params.append( param )
    
    # create list of the named_parameters of each section of the model
    model_named_parameters = list(model.named_parameters())
    transDownBlocks = list(model.transDownBlocks.named_parameters())
    transUpBlocks = list(model.transUpBlocks.named_parameters())
    denseBlocksDown = list(model.denseBlocksDown.named_parameters())
    denseBlocksUp = list(model.denseBlocksUp.named_parameters())
    bottleneck = list(model.bottleneck.named_parameters())

    # add names to each part of the list
    transDownBlocks = add_names(transDownBlocks, 'transDownBlocks')
    transUpBlocks = add_names(transUpBlocks, 'transUpBlocks')
    denseBlocksDown = add_names(denseBlocksDown, 'denseBlocksDown')
    denseBlocksUp = add_names(denseBlocksUp, 'denseBlocksUp')
    bottleneck = add_names(bottleneck, 'bottleneck')

    # now combine the transdown and dense blocks down layers
    dense_trans_Down = []
    current_block = 0
    for i, (name, param) in enumerate(denseBlocksDown):
        #print((i+1)%4 is 0, int(name.split('.')[2]) is 0)
        dense_trans_Down.append( (name, param) )
        
        if (i+1)%4 is 0 and int(name.split('.')[3])==(model.down_blocks[current_block]-1):
            #print(i, name)
            
            for j in range(4):
                idx = current_block*4 + j
                #print(idx)
                dense_trans_Down.append( transDownBlocks[idx])
                
            current_block +=1

    # throw error if layers are missing
    assert len(dense_trans_Down) == (len(denseBlocksDown) + len(transDownBlocks))

    # do the same for denseblocks up and transition up
    dense_trans_Up = []
    current_block = 0
    start_idx = 0
    for i, (name, param) in enumerate(transUpBlocks):
        dense_trans_Up.append( (name, param) )
        if (i+1)%2 is 0:
            for j in range(4 * model.up_blocks[current_block] ):
                idx = j + start_idx
                dense_trans_Up.append( denseBlocksUp[idx] )
            start_idx += (4 * model.up_blocks[current_block])
            current_block += 1

    # ensure up is the same as well
    assert len(dense_trans_Up) == (len(denseBlocksUp) + len(transUpBlocks))

    # now combine the first/last layers + the dense/trans layers + bottleneck
    sorted_param_list = model_named_parameters[0:2] + dense_trans_Down + bottleneck + dense_trans_Up + \
                    [model_named_parameters[-2]] + [model_named_parameters[-1]]

    # finally create a new parameter list with the sorted parameter names
    param_list_t_weights = []
    for (name, param), t_param in zip(sorted_param_list, no_bad_params):
        #ensure all parameters are the same size
        assert list(param.size()) == list(t_param.shape)

        #add tuple of name + tensor of theano weight
        # flip convolutional layers due to lasagne defaults
        #if len(param.size())==4:
        #    param_list_t_weights.append((name, torch.FloatTensor(t_param[:,:,::-1,::-1].copy()).cuda()))
        #else:
        param_list_t_weights.append((name, torch.FloatTensor(t_param).cuda()))

    # convert named list to a dict and load (didn't get the batch norm parameters apparently)
    param_odict_t_weights = dict(param_list_t_weights)
    model.load_state_dict(param_odict_t_weights, strict=False)

def __main__():
    # load model
    model = FCDenseNet103(n_classes=11)

    # load weights of saved model
    model_weights = np.load(args.theano_model_path)

    # now update weights of model
    load_theano_fcmodel(model, model_weights)

    model.cuda()

    normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)
    train_joint_transformer = transforms.Compose([
        joint_transforms.JointRandomCrop(224), # commented for fine-tuning
        joint_transforms.JointRandomHorizontalFlip()
        ])
    train_dset = camvid.CamVid(args.dataset_path, 'train',
        joint_transform=train_joint_transformer,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=1, shuffle=True)

    test_dset = camvid.CamVid(
        args.dataset_path, 'test', joint_transform=None,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dset, batch_size=1, shuffle=False)

    criterion = torch.nn.NLLLoss()

    optimizer = torch.optim.RMSprop(model.parameters(), lr = 1e-4, weight_decay = 1e-4)

    # update batch norm parameters
    for i in range(50):
        train_results = train(model, train_loader, optimizer, criterion)
        print('Train Loss/Accuracy/IOU: ', train_results)
        test_results = test(model, test_loader, criterion)  
        print('Test Loss/Accuracy/IOU: ', test_results)

__main__()


    






