from torchvision import transforms
from torch.utils.data import Dataset
from .data_utils import get_onehot
from .augmentation.randaugment import RandAugment
from .augmentation.eda import *

import torchvision
from PIL import Image
import numpy as np
import copy


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 img,
                 text,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
                 strong_transform=None,
                 onehot=False,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.img = img
        self.text = text
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot

        self.transform = transform
        # self.text_transform = get_only_chars()
        # self.text_stong_transform = eda()
        if self.is_ulb:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3, 5))
        else:
            self.strong_transform = strong_transform

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """

        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented images

        # img = Image.open(self.data[idx]).convert('RGB')
        img = self.img[idx]
        text = self.text[idx]
        
        if self.transform is None:
            return transforms.ToTensor()(img), text, target
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)
            text_w = get_only_chars(text)
            if not self.is_ulb:
                return idx, img_w, text, target
            else:
                if self.alg == 'fixmatch':
                    return idx, img_w, self.strong_transform(img), text, target
                elif self.alg == 'comatch':
                    return idx, img_w, self.strong_transform(img), self.strong_transform(img), \
                        text, eda(text),eda(text),target
                elif self.alg == 'flexmatch':
                    return idx, img_w, self.strong_transform(img), text, target
                elif self.alg == 'pimodel':
                    return idx, img_w, self.transform(img), text, target
                elif self.alg == 'pseudolabel':
                    return idx, img_w, text, target
                elif self.alg == 'vat':
                    return idx, img_w, text, target
                elif self.alg == 'meanteacher':
                    return idx, img_w, self.transform(img), text, target
                elif self.alg == 'uda':
                    return idx, img_w, self.strong_transform(img), text, target
                elif self.alg == 'mixmatch':
                    return idx, img_w, self.transform(img), text, target
                elif self.alg == 'remixmatch':
                    rotate_v_list = [0, 90, 180, 270]
                    rotate_v1 = np.random.choice(rotate_v_list, 1).item()
                    img_s1 = self.strong_transform(img)
                    img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
                    img_s2 = self.strong_transform(img)
                    return idx, img_w, img_s1, img_s2, img_s1_rot, rotate_v_list.index(rotate_v1), text, target
                elif self.alg == 'fullysupervised':
                    return idx
                elif self.alg == 'main':
                    return idx, img_w, self.strong_transform(img), text, target

    def __len__(self):
        return len(self.img)
