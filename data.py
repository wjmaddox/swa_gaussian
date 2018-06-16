import numpy as np
import torch
import torchvision
import os
import regression_data

c10_classes = np.array([
    [0, 1, 2, 8, 9],
    [3, 4, 5, 6, 7]
], dtype=np.int32)

def loaders(dataset, path, batch_size, num_workers, transform_train, transform_test, use_validation=True, val_size=5000, split_classes=None, shuffle_train=True):

    regression_problem = False
    try:
        ds = getattr(torchvision.datasets, dataset)
    except:
        if dataset == 'toy_regression':
            ds = regression_data.generate_toy_problem
            regression_problem = True
    path = os.path.join(path, dataset.lower())
    train_set = ds(root=path, train=True, download=True, transform=transform_train)


    if use_validation:
        print("Using train (45000) + validation (5000)")
        train_set.train_data = train_set.train_data[:-5000]
        train_set.train_labels = train_set.train_labels[:-5000]

        test_set = ds(root=path, train=True, download=True, transform=transform_test)
        test_set.train = False
        test_set.test_data = test_set.train_data[-5000:]
        test_set.test_labels = test_set.train_labels[-5000:]
        delattr(test_set, 'train_data')
        delattr(test_set, 'train_labels')
    else:
        print('You are going to run models on the test set. Are you sure?')
        test_set = ds(path, train=False, download=True, transform=transform_test)

    if split_classes is not None:
        assert dataset == 'CIFAR10'
        assert split_classes in {0, 1}

        print('Using classes:', end='')
        print(c10_classes[split_classes])
        train_mask = np.isin(train_set.train_labels, c10_classes[split_classes])
        train_set.train_data = train_set.train_data[train_mask, :]
        train_set.train_labels = np.array(train_set.train_labels)[train_mask]
        train_set.train_labels = np.where(train_set.train_labels[:, None] == c10_classes[split_classes][None, :])[1].tolist()
        print('Train: %d/%d' % (train_set.train_data.shape[0], train_mask.size))

        test_mask = np.isin(test_set.test_labels, c10_classes[split_classes])
        test_set.test_data = test_set.test_data[test_mask, :]
        test_set.test_labels = np.array(test_set.test_labels)[test_mask]
        test_set.test_labels = np.where(test_set.test_labels[:, None] == c10_classes[split_classes][None, :])[1].tolist()
        print('Test: %d/%d' % (test_set.test_data.shape[0], test_mask.size))

    num_classes = max(train_set.train_labels) + 1
    if regression_problem:
        num_classes = 0

    return \
        {
            'train': torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True and shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            ),
        }, \
        num_classes

