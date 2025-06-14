import torch
import torchvision
from torchvision import datasets
from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch.distributed as dist
import numpy as np
import json
import os

from datasets.DistributedProxySampler import DistributedProxySampler


def split_ssl_data(args, img, text, img_label, text_label, label, num_labels, num_classes, init_len=0, index=None, include_lb_to_ulb=True):
    """
    data & target is splitted into labeled and unlabeld data.
    
    Args
        index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeld data
    """
    # data, target = np.array(data), np.array(target)
    img, text, img_label, text_label, label = np.array(img), np.array(text), np.array(img_label), np.array(text_label), np.array(label)
    lb_img, lb_text, lbs, lb_idx = sample_labeled_data(args, img, text, label, num_labels, num_classes, args.dataset, index)
    ulb_idx = np.array(sorted(list(set(range(len(img))) - set(lb_idx))))  # unlabeled_data index of data
    if include_lb_to_ulb:
        return lb_img, lb_text, lbs, img, text, label
    else:
        return lb_img, lb_text, lbs, img[ulb_idx], text[ulb_idx], label[ulb_idx]


def sample_labeled_data(args, img, text, label,
                        num_labels, num_classes,
                        dataset, index=None, name=None):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    assert num_labels % num_classes == 0
    if not index is None:
        index = np.array(index, dtype=np.int32)
        return img[index], text[index], label[index], index

    dump_file = 'sampled_label_idx_' + str(dataset) + '_' + str(num_labels) + '.npy'

    dump_path = os.path.join(args.save_dir, args.save_name, dump_file)

    if os.path.exists(dump_path):
        lb_idx = np.load(dump_path)
        lb_img = img[lb_idx]
        lb_text = text[lb_idx]
        lbs = label[lb_idx]
        return lb_img, lb_text, lbs, lb_idx

    samples_per_class = int(num_labels / num_classes)

    lb_img = []
    lb_text = []
    lbs = []
    lb_idx = []
    for c in range(num_classes):
        idx = np.where(label == c)[0]
        idx = np.random.choice(idx, samples_per_class, False)
        lb_idx.extend(idx)

        lb_img.extend(img[idx])
        lb_text.extend(text[idx])
        lbs.extend(label[idx])

    np.save(dump_path, np.array(lb_idx))

    return np.array(lb_img), np.array(lb_text), np.array(lbs), np.array(lb_idx)


def get_sampler_by_name(name):
    '''
    get sampler in torch.utils.data.sampler by name
    '''
    sampler_name_list = sorted(name for name in torch.utils.data.sampler.__dict__
                               if not name.startswith('_') and callable(sampler.__dict__[name]))
    try:
        if name == 'DistributedSampler':
            return torch.utils.data.distributed.DistributedSampler
        else:
            return getattr(torch.utils.data.sampler, name)
    except Exception as e:
        print(repr(e))
        print('[!] select sampler in:\t', sampler_name_list)


# def get_data_loader(dset,
#                     batch_size=None,
#                     shuffle=False,
#                     num_workers=4,
#                     pin_memory=False,
#                     data_sampler=None,
#                     replacement=True,
#                     num_epochs=None,
#                     num_iters=None,
#                     generator=None,
#                     drop_last=True,
#                     distributed=False):
def get_data_loader(dset,
                    batch_size=None,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False,
                    data_sampler=None,
                    replacement=True,
                    num_epochs=None,
                    num_iters=None,
                    generator=None,
                    drop_last=True):
    """
    get_data_loader returns torch.utils.data.DataLoader for a Dataset.
    All arguments are comparable with those of pytorch DataLoader.
    However, if distributed, DistributedProxySampler, which is a wrapper of data_sampler, is used.
    
    Args
        num_epochs: total batch -> (# of batches in dset) * num_epochs 
        num_iters: total batch -> num_iters
    """

    assert batch_size is not None

    if data_sampler is None:
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory)

    else:
        if isinstance(data_sampler, str):
            data_sampler = get_sampler_by_name(data_sampler)

        # if distributed:
        #     assert dist.is_available()
        #     num_replicas = dist.get_world_size()
        # else:
        #     num_replicas = 1
        
        num_replicas = 1

        if (num_epochs is not None) and (num_iters is None):
            num_samples = len(dset) * num_epochs
        elif (num_epochs is None) and (num_iters is not None):
            num_samples = batch_size * num_iters * num_replicas
        else:
            num_samples = len(dset)

        if data_sampler.__name__ == 'RandomSampler':
            data_sampler = data_sampler(dset, replacement, num_samples, generator)
        else:
            raise RuntimeError(f"{data_sampler.__name__} is not implemented.")

        # if distributed:
        #     '''
        #     Different with DistributedSampler, 
        #     the DistribuedProxySampler does not shuffle the data (just wrapper for dist).
        #     '''
        #     data_sampler = DistributedProxySampler(data_sampler)

        batch_sampler = BatchSampler(data_sampler, batch_size, drop_last)
        return DataLoader(dset, batch_sampler=batch_sampler,
                          num_workers=num_workers, pin_memory=pin_memory)


def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot
