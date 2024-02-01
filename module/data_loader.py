import timm
from timm.data import create_transform
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from module.custom_dataset import JigsawDataset
import pandas as pd

def build_transform(is_train):
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size = (384, 384),
            is_training = True,
            color_jitter = 0.3,
            auto_augment = 'rand-m9-mstd0.5-inc1',
            interpolation= 'bicubic',
            re_prob= 0.25,
            re_mode= 'pixel',
            re_count= 1,
        )
        return transform

    t = []
    t.append(transforms.Resize((384,384), interpolation=3))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def get_dataframe(data_path='./data', train_size=-100):
    train_csv = pd.read_csv(data_path+'/train.csv')
    test_csv = pd.read_csv(data_path+'/test.csv')
    
    train_df = train_csv.iloc[:train_size]    # train dataframe
    valid_df = train_csv.iloc[train_size:]    # valid dataframe
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