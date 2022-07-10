import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
import collections
import dataset_train 
import os

def get_dataset(opt):
    return dataset_train.read_data(opt)


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler

def print_label_distribution(dataset, num):
    length=len(dataset)
    print('>>> Size of dataset:', length)
    print(len(dataset[0]))
    print(dataset[0][0].shape, dataset[0][1])
    l=[dataset[np.random.randint(0, length)][1] for _ in range(num)]
    d=collections.Counter(l)
    k=sorted(d.keys())
    print('>>> Label distribution for %d random samples in original dataset:    '%(num))
    for i in range(len(k)):
        print('>>>', k[i], ':', d[k[i]])


def patch_collate_train(batch):
    input_img=torch.stack([item[0] for item in batch], dim=0)
    cropped_img=torch.stack([item[1] for item in batch], dim=0)
    target=torch.tensor([item[2] for item in batch])
    scale=torch.stack([item[3] for item in batch], dim=0)
    return [input_img, cropped_img, target, scale]

def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    print('>>> Size of dataset: ', len(dataset))
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              collate_fn=patch_collate_train,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader

def patch_collate_test(batch):
    input_img=[item[0] for item in batch]
    cropped_img=torch.stack([item[1] for item in batch], dim=0)
    target=torch.tensor([item[2] for item in batch])
    scale=torch.stack([item[3] for item in batch], dim=0)
    filename=[item[4] for item in batch]
    return [input_img, cropped_img, target, scale, filename]

def create_dataloader_test(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    print('>>> Size of dataset: ', len(dataset))
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader_test = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              collate_fn=patch_collate_test,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader_test
