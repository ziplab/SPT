import os
import torch

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from lib.utils import read_json
from collections import Counter
import torchvision as tv
import numpy as np


class general_dataset(ImageFolder):
    def __init__(self, root, train=True, transform=None, test=False):
        self.dataset_root = root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform

        if not test:
            train_list_path = os.path.join(self.dataset_root, 'train800.txt')
            test_list_path = os.path.join(self.dataset_root, 'val200.txt')
        else:
            train_list_path = os.path.join(self.dataset_root, 'train800val200.txt')
            test_list_path = os.path.join(self.dataset_root, 'test.txt')

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))


class general_dataset_few_shot(ImageFolder):
    def __init__(self, root, dataset,train=True, transform=None, target_transform=None, mode=None,is_individual_prompt=False,shot=2,seed=0,**kwargs):
        self.dataset_root = root
        self.dataset = dataset.replace('-FS','')
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform
        
        if mode == 'super' and is_individual_prompt==False:
            train_list_path = os.path.join(self.dataset_root, 'annotations/train_meta.list.num_shot_'+str(shot)+'.seed_'+str(seed))
        elif mode == 'search' and is_individual_prompt==False:
            if 'imagenet' in root:
                train_list_path = os.path.join(self.dataset_root, 'annotations/unofficial_val_list_4_shot16seed0')
            else:
                train_list_path = os.path.join(self.dataset_root, 'annotations/val_meta.list')
        else:
            if 'imagenet' in root and self.dataset != 'imagenet':
                train_list_path = os.path.join(self.dataset_root, 'annotations/val_meta.list')
            else:
                train_list_path = os.path.join(self.dataset_root, 'annotations/train_meta.list.num_shot_'+str(shot)+'.seed_'+str(seed))

        if mode == 'search':
            test_list_path = os.path.join(self.dataset_root, 'annotations/train_meta.list.num_shot_1.seed_0')           
        elif 'imagenet' in root:
            test_list_path = os.path.join(self.dataset_root, 'annotations/val_meta.list')
        else:
            test_list_path = os.path.join(self.dataset_root, 'annotations/test_meta.list')

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.rsplit(' ',1)[0]
                    label = int(line.rsplit(' ',1)[1])
                    if 'stanford_cars' in root or ('imagenet' in root and 'imagenet' != self.dataset):
                        self.samples.append((os.path.join(root,img_name), label))
                    elif 'imagenet' == self.dataset:
                        self.samples.append((os.path.join(root+'/train',img_name), label))
                    else:
                        self.samples.append((os.path.join(root+'/images',img_name), label))
                    
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.rsplit(' ',1)[0]
                    label = int(line.rsplit(' ',1)[1])
                    if 'stanford_cars' in root or ('imagenet' in root and 'imagenet' != self.dataset):
                        self.samples.append((os.path.join(root,img_name), label))
                    elif 'imagenet' == self.dataset:
                        if mode == 'search':
                            self.samples.append((os.path.join(root+'/train',img_name), label))
                        else:
                            self.samples.append((os.path.join(root+'/val',img_name), label))
                    else:
                        self.samples.append((os.path.join(root+'/images',img_name), label))


# From visual prompt tuning
class JSONDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):

        assert split in {
            "train",
            "val",
            "trainval",
            "test",
        }, "Split '{}' not supported for {} dataset".format(
            split, cfg.data_set)
        print("Constructing {} dataset {}...".format(
            cfg.data_set, split))

        self.cfg = cfg
        self._split = split
        self.name = cfg.data_set
        self.data_dir = cfg.data_path
        self.data_percentage = cfg.data_percentage
        self._construct_imdb(cfg)
        self.transform = build_transform((split == "train"), cfg)

    def get_anno(self):
        anno_path = os.path.join(self.data_dir, "{}.json".format(self._split))
        if "train" in self._split:
            if self.data_percentage < 1.0:
                anno_path = os.path.join(
                    self.data_dir,
                    "{}_{}.json".format(self._split, self.data_percentage)
                )
        print('annotation path: ', anno_path)
        assert os.path.exists(anno_path), "{} dir not found".format(anno_path)

        return read_json(anno_path)

    def get_imagedir(self):
        raise NotImplementedError()

    def _construct_imdb(self, cfg):
        """Constructs the imdb."""

        img_dir = self.get_imagedir()
        assert os.path.exists(img_dir), "{} dir not found".format(img_dir)

        anno = self.get_anno()
        # Map class ids to contiguous ids
        self._class_ids = sorted(list(set(anno.values())))
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}

        # Construct the image db
        self._imdb = []
        for img_name, cls_id in anno.items():
            cont_id = self._class_id_cont_id[cls_id]
            im_path = os.path.join(img_dir, img_name)
            self._imdb.append({"im_path": im_path, "class": cont_id})

        print("Number of images: {}".format(len(self._imdb)))
        print("Number of classes: {}".format(len(self._class_ids)))

    def get_info(self):
        num_imgs = len(self._imdb)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self._split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num

        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        # Load the image
        im = tv.datasets.folder.default_loader(self._imdb[index]["im_path"])
        label = self._imdb[index]["class"]
        im = self.transform(im)
        if self._split == "train":
            index = index
        else:
            index = f"{self._split}{index}"
        sample = {
            "image": im,
            "label": label,
        }
        return sample["image"], sample["label"]

    def __len__(self):
        return len(self._imdb)


class CUB200Dataset(JSONDataset):
    """CUB_200 dataset."""

    def __init__(self, cfg, split):
        super(CUB200Dataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")


class CarsDataset(JSONDataset):
    """stanford-cars dataset."""

    def __init__(self, cfg, split):
        super(CarsDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return self.data_dir


class DogsDataset(JSONDataset):
    """stanford-dogs dataset."""

    def __init__(self, cfg, split):
        super(DogsDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "Images")


class FlowersDataset(JSONDataset):
    """flowers dataset."""

    def __init__(self, cfg, split):
        super(FlowersDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return self.data_dir


class NabirdsDataset(JSONDataset):
    """Nabirds dataset."""

    def __init__(self, cfg, split):
        super(NabirdsDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")


def build_dataset(is_train, args, folder_name=None, imagenet=None):
    transform = build_transform(is_train, args)

    if imagenet or args.data_set == 'IMNET':
        root = os.path.join('data/imagenet', 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'CARS':
        dataset = Cars196(args.data_path, train=is_train, transform=transform)
        nb_classes = 196
    elif args.data_set == 'PETS':
        dataset = Pets(args.data_path, train=is_train, transform=transform)
        nb_classes = 37
    elif args.data_set == 'FLOWERS':
        dataset = Flowers(args.data_path, train=is_train, transform=transform)
        nb_classes = 102
    elif args.data_set == 'EVO_IMNET':
        root = os.path.join(args.data_path, folder_name)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'clevr_count':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 8
    elif args.data_set == 'diabetic_retinopathy':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 5
    elif args.data_set == 'dsprites_loc':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 16
    elif args.data_set == 'dtd':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 47
    elif args.data_set == 'kitti':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 4
    elif args.data_set == 'oxford_iiit_pet':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 37
    elif args.data_set == 'resisc45':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 45
    elif args.data_set == 'smallnorb_ele':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 9
    elif args.data_set == 'svhn':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 10
    elif args.data_set == 'cifar':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 100
    elif args.data_set == 'clevr_dist':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 6
    elif args.data_set == 'caltech101':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 102
    elif args.data_set == 'dmlab':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 6
    elif args.data_set == 'dsprites_ori':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 16
    elif args.data_set == 'eurosat':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 10
    elif args.data_set == 'oxford_flowers102':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 102
    elif args.data_set == 'patch_camelyon':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 2
    elif args.data_set == 'smallnorb_azi':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 18
    elif args.data_set == 'sun397':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform, test=args.test)
        nb_classes = 397
    # FGVC
    elif args.data_set == 'stanforddog':

        if is_train:
            split = "train"
        elif args.test:
            split = "test"
        else:
            split = "val"

        dataset = DogsDataset(args, split)
        nb_classes = 120
    elif args.data_set == 'stanfordcar':

        if is_train:
            split = "train"
        elif args.test:
            split = "test"
        else:
            split = "val"

        dataset = CarsDataset(args, split)
        nb_classes = 196
    elif args.data_set == 'oxfordflower':

        if is_train:
            split = "train"
        elif args.test:
            split = "test"
        else:
            split = "val"

        dataset = FlowersDataset(args, split)
        nb_classes = 102
    elif args.data_set == 'nabirds':

        if is_train:
            split = "train"
        elif args.test:
            split = "test"
        else:
            split = "val"

        dataset = NabirdsDataset(args, split)
        nb_classes = 555
    elif args.data_set == 'cub':
        if is_train:
            split = "train"
        elif args.test:
            split = "test"
        else:
            split = "val"

        dataset = CUB200Dataset(args, split)
        nb_classes = 200
    elif '-FS' in args.data_set:
        dataset = general_dataset_few_shot(args.data_path, args.data_set,train=is_train, transform=transform,mode=args.mode,is_individual_prompt=is_individual_prompt,shot=args.few_shot_shot,seed=args.few_shot_seed)
        if 'stanford_cars' in args.data_set:
            nb_classes = 196
        elif 'oxford_flowers' in args.data_set:
            nb_classes = 102
        elif 'food-101' in args.data_set:
            nb_classes = 101
        elif 'oxford_pets'in args.data_set:
            nb_classes = 37
        elif 'fgvc_aircraft' in args.data_set:
            nb_classes = 100
        elif 'imagenet' in args.data_set:
            nb_classes = 1000

    return dataset, nb_classes


def build_transform(is_train, args):

    if not args.no_aug and is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            interpolation=args.train_interpolation,
        )
        return transform

    t = []

    if args.direct_resize:
        # For VTAB-1k
        size = args.input_size
        t.append(
            transforms.Resize((size, size), interpolation=3)  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    else:
        # For FGVC, also resize and flip
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize((size,size), interpolation=3)  # to maintain same ratio w.r.t. 224 images
        )

        if is_train:
            t.append(transforms.RandomCrop(args.input_size))
            t.append(transforms.RandomHorizontalFlip(0.5))
        else:
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    if args.inception:
        t.append(transforms.Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD))
    else:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

