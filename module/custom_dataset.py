import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image

class JigsawDataset(Dataset):
    def __init__(self, df, data_path, mode='train', transform=None):
        self.df = df
        self.data_path = data_path
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.mode == 'train':
            row = self.df.iloc[idx]
            image = read_image(os.path.join(self.data_path, row['img_path']))
            shuffle_order = row[[str(i) for i in range(1, 17)]].values-1
            image = self.reset_image(image, shuffle_order)
            image = Image.fromarray(image)

            if self.transform:
                image = self.transform(image)
            return image

        elif self.mode == 'test':
            row = self.df.iloc[idx]
            image = Image.open(os.path.join(self.data_path, row['img_path']))

            if self.transform:
                image = self.transform(image)
            return image

    def reset_image(self, image, shuffle_order):
        c, h, w = image.shape
        block_h, block_w = h//4, w//4

        image_src = [[0 for _ in range(4)] for _ in range(4)]
        for idx, order in enumerate(shuffle_order):
            h_idx, w_idx = divmod(order,4)
            h_idx_shuffle, w_idx_shuffle = divmod(idx, 4)
            image_src[h_idx][w_idx] = image[:, block_h * h_idx_shuffle : block_h * (h_idx_shuffle+1), block_w * w_idx_shuffle : block_w * (w_idx_shuffle+1)]
        
        image_src = np.concatenate([np.concatenate(image_row, -1) for image_row in image_src], -2)
        return image_src.transpose(1, 2, 0)