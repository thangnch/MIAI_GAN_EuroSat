from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import os


class dataset(Dataset):
    def __init__(self, batch_size, images_paths, targets, img_size=64, path_target='data/targets',
                 path_input='data/inputs'):
        self.batch_size = batch_size
        self.path_input = path_input
        self.path_target = path_target
        self.img_size = img_size
        self.images_paths = images_paths
        self.targets = targets
        self.len = len(self.images_paths) // batch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.batch_im = [self.images_paths[idx * self.batch_size:(idx + 1) * self.batch_size] for idx in
                         range(self.len)]

        self.batch_t = [self.targets[idx * self.batch_size:(idx + 1) * self.batch_size] for idx in range(self.len)]


    def __getitem__(self, idx):

        pred = torch.stack([
            self.transform(Image.open(os.path.join(self.path_input, file_name)))
            for file_name in self.batch_im[idx]
        ])
        target = torch.stack([
            self.transform(Image.open(os.path.join(self.path_target, file_name)))
            for file_name in self.batch_t[idx]
        ])

        return pred, target

    def __len__(self):
        return self.len
