import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms


class ChestXrayDataset(Dataset):
    def __init__(self, data_dir, csv_file, split_list_file, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            split_list_file (string): train_val_list.txt or test_list.txt.
            transform (callable, optional): Augmentations
        """
        self.data_dir = data_dir
        self.transform = transform

        self.df = pd.read_csv(csv_file)

        # which images will be in the training process
        with open(split_list_file, 'r') as f:
            valid_filenames = set([line.strip() for line in f.readlines()])

        self.df = self.df[self.df['Image Index'].isin(valid_filenames)].reset_index(drop=True)

        self.labels = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
            'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
            'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
        ]

        # One-hot encoding
        for label in self.labels:
            self.df[label] = self.df['Finding Labels'].map(lambda x: 1.0 if label in x else 0.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Image Index']
        img_path = os.path.join(self.data_dir, img_name)

        # from 1 to 3 channels
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))

        labels = self.df.iloc[idx][self.labels].values.astype('float32')
        labels = torch.tensor(labels)

        # Using transform(resize, normalize, etc.)
        if self.transform:
            image = self.transform(image)

        return image, labels


if __name__ == '__main__':
    print("Testing ChestXrayDataset...")

    # simple transformation
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ChestXrayDataset(
        data_dir=r'C:\Users\srbuh\Desktop\Medical_AI_Diagnosis\data\images',
        csv_file=r'C:\Users\srbuh\Desktop\Medical_AI_Diagnosis\data\Data_Entry_2017.csv',
        split_list_file=r'C:\Users\srbuh\Desktop\Medical_AI_Diagnosis\data\test_list.txt',
        transform=test_transform )

    print(f"Length of dataset: {len(dataset)}")

    # 1st image for testing
    img, lab = dataset[0]

    print(f"size of img (Tensor): {img.size()}")  # should be [3, 224, 224]
    print(f"Labels: {lab}")