import numpy as np
import torch
import torchvision
import os

from .camvid import CamVid

c10_classes = np.array([[0, 1, 2, 8, 9], [3, 4, 5, 6, 7]], dtype=np.int32)


def camvid_loaders(
    path,
    batch_size,
    num_workers,
    transform_train,
    transform_test,
    use_validation,
    val_size,
    shuffle_train=True,
    joint_transform=None,
    ft_joint_transform=None,
    ft_batch_size=1,
    **kwargs
):

    # load training and finetuning datasets
    print(path)
    train_set = CamVid(
        root=path,
        split="train",
        joint_transform=joint_transform,
        transform=transform_train,
        **kwargs
    )
    ft_train_set = CamVid(
        root=path,
        split="train",
        joint_transform=ft_joint_transform,
        transform=transform_train,
        **kwargs
    )

    val_set = CamVid(
        root=path, split="val", joint_transform=None, transform=transform_test, **kwargs
    )
    test_set = CamVid(
        root=path,
        split="test",
        joint_transform=None,
        transform=transform_test,
        **kwargs
    )

    num_classes = 11  # hard coded labels ehre

    return (
        {
            "train": torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "fine_tune": torch.utils.data.DataLoader(
                ft_train_set,
                batch_size=ft_batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "val": torch.utils.data.DataLoader(
                val_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "test": torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        },
        num_classes,
    )


def svhn_loaders(
    path,
    batch_size,
    num_workers,
    transform_train,
    transform_test,
    use_validation,
    val_size,
    shuffle_train=True,
):
    train_set = torchvision.datasets.SVHN(
        root=path, split="train", download=True, transform=transform_train
    )

    if use_validation:
        test_set = torchvision.datasets.SVHN(
            root=path, split="train", download=True, transform=transform_test
        )
        train_set.data = train_set.data[:-val_size]
        train_set.labels = train_set.labels[:-val_size]

        test_set.data = test_set.data[-val_size:]
        test_set.labels = test_set.labels[-val_size:]

    else:
        print("You are going to run models on the test set. Are you sure?")
        test_set = torchvision.datasets.SVHN(
            root=path, split="test", download=True, transform=transform_test
        )

    num_classes = 10

    return (
        {
            "train": torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True and shuffle_train,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "test": torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        },
        num_classes,
    )


def loaders(
    dataset,
    path,
    batch_size,
    num_workers,
    transform_train,
    transform_test,
    use_validation=True,
    val_size=5000,
    split_classes=None,
    shuffle_train=True,
    **kwargs
):

    if dataset == "CamVid":
        return camvid_loaders(
            path,
            batch_size=batch_size,
            num_workers=num_workers,
            transform_train=transform_train,
            transform_test=transform_test,
            use_validation=use_validation,
            val_size=val_size,
            **kwargs
        )

    path = os.path.join(path, dataset.lower())

    ds = getattr(torchvision.datasets, dataset)

    if dataset == "SVHN":
        return svhn_loaders(
            path,
            batch_size,
            num_workers,
            transform_train,
            transform_test,
            use_validation,
            val_size,
        )
    else:
        ds = getattr(torchvision.datasets, dataset)

    if dataset == "STL10":
        train_set = ds(
            root=path, split="train", download=True, transform=transform_train
        )
        num_classes = 10
        cls_mapping = np.array([0, 2, 1, 3, 4, 5, 7, 6, 8, 9])
        train_set.labels = cls_mapping[train_set.labels]
    else:
        train_set = ds(root=path, train=True, download=True, transform=transform_train)
        num_classes = max(train_set.targets) + 1

    if use_validation:
        print(
            "Using train ("
            + str(len(train_set.data) - val_size)
            + ") + validation ("
            + str(val_size)
            + ")"
        )
        train_set.data = train_set.data[:-val_size]
        train_set.targets = train_set.targets[:-val_size]

        test_set = ds(root=path, train=True, download=True, transform=transform_test)
        test_set.train = False
        test_set.data = test_set.data[-val_size:]
        test_set.targets = test_set.targets[-val_size:]
        # delattr(test_set, 'data')
        # delattr(test_set, 'targets')
    else:
        print("You are going to run models on the test set. Are you sure?")
        if dataset == "STL10":
            test_set = ds(
                root=path, split="test", download=True, transform=transform_test
            )
            test_set.labels = cls_mapping[test_set.labels]
        else:
            test_set = ds(
                root=path, train=False, download=True, transform=transform_test
            )

    if split_classes is not None:
        assert dataset == "CIFAR10"
        assert split_classes in {0, 1}

        print("Using classes:", end="")
        print(c10_classes[split_classes])
        train_mask = np.isin(train_set.targets, c10_classes[split_classes])
        train_set.data = train_set.data[train_mask, :]
        train_set.targets = np.array(train_set.targets)[train_mask]
        train_set.targets = np.where(
            train_set.targets[:, None] == c10_classes[split_classes][None, :]
        )[1].tolist()
        print("Train: %d/%d" % (train_set.data.shape[0], train_mask.size))

        test_mask = np.isin(test_set.targets, c10_classes[split_classes])
        print(test_set.data.shape, test_mask.shape)
        test_set.data = test_set.data[test_mask, :]
        test_set.targets = np.array(test_set.targets)[test_mask]
        test_set.targets = np.where(
            test_set.targets[:, None] == c10_classes[split_classes][None, :]
        )[1].tolist()
        print("Test: %d/%d" % (test_set.data.shape[0], test_mask.size))

    return (
        {
            "train": torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True and shuffle_train,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "test": torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        },
        num_classes,
    )
