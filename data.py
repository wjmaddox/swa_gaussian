import torch
import torchvision
import os

def loaders(dataset, path, batch_size, num_workers, transform_train, transform_test, use_validation=True):
    ds = getattr(torchvision.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    train_set = ds(path, train=True, download=True, transform=transform_train)

    if use_validation:
        print("Using train (45000) + validation (5000)")
        train_set.train_data = train_set.train_data[:-5000]
        train_set.test_labels = train_set.train_labels[:-5000]

        test_set = ds(path, train=True, download=True, transform=transform_test)
        test_set.train = False
        test_set.test_data = test_set.train_data[-5000:]
        test_set.test_labels = test_set.train_labels[-5000:]
        delattr(test_set, 'train_data')
        delattr(test_set, 'train_labels')
    else:
        print('You are going to run models on the test set. Are you sure?')
        test_set = ds(path, train=False, download=True, transform=transform_test)
    return \
        {
            'train': torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
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
        max(train_set.train_labels) + 1

