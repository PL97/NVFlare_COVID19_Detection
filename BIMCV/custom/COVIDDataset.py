import numpy as np
from PIL import Image
import pandas as pd
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import torch
import os


class COVIDDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        assert mode in ['train', 'val'], "invalid mode"
        df = pd.read_csv(os.path.join(root_dir, "{}.csv".format(mode)))
        self.path = list(df['square path'])
        self.label = df['label']
        self.root_dir = root_dir
        self.mode = mode

        self.transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop((224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}

    def __getitem__(self, index):
        p = self.path[index]
        img = Image.open(p).convert("RGB")
        label = self.label[index]
        img = self.transform[self.mode](img)
        return img, label

    def __len__(self):
        return len(self.path)

    
if __name__ == "__main__":

    # pass    

    # # test case
    dl = COVIDDataset(root_dir="/datadrive/new_png/public/clean/1/", mode='train')
    for img, label in dl:
        print(img.shape, label)
        asdf


