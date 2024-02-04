import timm
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from module.custom_dataset import JigsawDataset
import pandas as pd

def build_transform(is_train):
    resize_transform = transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC)
    normalize_transform = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    if is_train:
        # Apply milder augmentation for training
        train_transform = transforms.Compose([
            resize_transform,
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            normalize_transform
        ])
        return train_transform

    # For validation, apply only resizing and normalization
    valid_transform = transforms.Compose([
        resize_transform,
        transforms.ToTensor(),
        normalize_transform
    ])
    return valid_transform

def get_dataframe(data_path='./data', train_size=-1, test_size=-1):
    train_csv = pd.read_csv(data_path+'/train.csv')
    test_csv = pd.read_csv(data_path+'/test.csv')
    
    if train_size != -1:
        train_csv = train_csv.iloc[:train_size]
        train_df = train_csv.iloc[:int(train_size * 0.8)]    # train dataframe
        valid_df = train_csv.iloc[int(train_size * 0.8):]    # valid dataframe
    else:
        train_df = train_csv.iloc[:int(len(train_csv) * 0.8)]    # train dataframe
        valid_df = train_csv.iloc[int(len(train_csv) * 0.8):]    # valid dataframe 

    if test_size != -1:
        test_df = test_csv.iloc[:test_size]
    else:
        test_df = test_csv.iloc[:]
    for i in range(1, 17): test_df[str(i)] = i

    return train_df, valid_df, test_df

def get_dataset(df, data_path='./data', is_train=True):
    if is_train:
        mode = "train"
    else:
        mode = "test"
    
    dataset = JigsawDataset(df = df,
                            data_path=data_path,
                            mode = mode,
                            transform = build_transform(is_train)
    )
    return dataset

def get_dataset_test(df, data_path='./data'):
    dataset = JigsawDataset(df = df,
                            data_path=data_path,
                            mode = "test",
                            transform = build_transform(False)
    )
    return dataset