import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.flip = 'flip' in self.feature_dir

        aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        # self.feature_files = sorted(os.listdir(feature_dir))
        # self.label_files = sorted(os.listdir(label_dir))
        # TODO: make it configurable
        self.feature_files = [f"{i}.npy" for i in range(1281167)]
        self.label_files = [f"{i}.npy" for i in range(1281167)]

    def __len__(self):
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir
                   
        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        features = np.load(os.path.join(feature_dir, feature_file))
        if self.flip:
            aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
            features = features[:, aug_idx]
        labels = np.load(os.path.join(label_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


class CustomDatasetDualCode(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.flip = 'flip' in self.feature_dir

        aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        self.feature_files_vis = [f"{i}_vis.npy" for i in range(1281167)]
        self.feature_files_sem = [f"{i}_sem.npy" for i in range(1281167)]
        self.label_files = [f"{i}.npy" for i in range(1281167)]

    def __len__(self):
        assert len(self.feature_files_vis) == len(self.feature_files_sem) == len(self.label_files), \
                "Number of feature files and label files should be same"
        return len(self.feature_files_vis)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir

        feature_file_vis = self.feature_files_vis[idx]
        feature_file_sem = self.feature_files_sem[idx]
        label_file = self.label_files[idx]

        features_vis = np.load(os.path.join(feature_dir, feature_file_vis))
        features_sem = np.load(os.path.join(feature_dir, feature_file_sem))
        if self.flip:
            aug_idx = torch.randint(low=0, high=features_vis.shape[1], size=(1,)).item()
            features_vis = features_vis[:, aug_idx]
            features_sem = features_sem[:, aug_idx]
        labels = np.load(os.path.join(label_dir, label_file))
        return torch.from_numpy(features_vis), torch.from_numpy(features_sem), torch.from_numpy(labels)


class CustomDatasetTripleCode(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.flip = 'flip' in self.feature_dir

        aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        self.feature_files_vis = [f"{i}_vis.npy" for i in range(1281167)]
        self.feature_files_sem_mid = [f"{i}_sem_mid.npy" for i in range(1281167)]
        self.feature_files_sem_high = [f"{i}_sem_high.npy" for i in range(1281167)]
        self.label_files = [f"{i}.npy" for i in range(1281167)]

    def __len__(self):
        assert len(self.feature_files_vis) == len(self.feature_files_sem_mid) == \
               len(self.feature_files_sem_high) == len(self.label_files), \
                "Number of feature files and label files should be same"
        return len(self.feature_files_vis)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir

        feature_file_vis = self.feature_files_vis[idx]
        feature_file_sem_mid = self.feature_files_sem_mid[idx]
        feature_file_sem_high = self.feature_files_sem_high[idx]
        label_file = self.label_files[idx]

        features_vis = np.load(os.path.join(feature_dir, feature_file_vis))
        features_sem_mid = np.load(os.path.join(feature_dir, feature_file_sem_mid))
        features_sem_high = np.load(os.path.join(feature_dir, feature_file_sem_high))
        if self.flip:
            aug_idx = torch.randint(low=0, high=features_vis.shape[1], size=(1,)).item()
            features_vis = features_vis[:, aug_idx]
            features_sem_mid = features_sem_mid[:, aug_idx]
            features_sem_high = features_sem_high[:, aug_idx]
        labels = np.load(os.path.join(label_dir, label_file))
        return torch.from_numpy(features_vis), torch.from_numpy(features_sem_mid), \
               torch.from_numpy(features_sem_high), torch.from_numpy(labels)


def build_imagenet(args, transform):
    return ImageFolder(args.data_path, transform=transform)


def build_imagenet_code(args):
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    return CustomDataset(feature_dir, label_dir)


def build_imagenet_dual_code(args):
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    return CustomDatasetDualCode(feature_dir, label_dir)


def build_imagenet_triple_code(args):
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    return CustomDatasetTripleCode(feature_dir, label_dir)

