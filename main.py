from module.data_download import download_and_unzip, download_lib
if __name__ == "__main__":
    download_lib()
    download_and_unzip()

import yaml
with open('./config.yaml') as f:
    config = yaml.safe_load(f)

from module.data_loader import get_dataframe, get_dataset
train_df, valid_df, test_df = get_dataframe(config["data_path"], config["train_size"])

train_dataset = get_dataset(train_df, config["data_path"], True)
valid_dataset = get_dataset(valid_df, config["data_path"], False)

import torch
from torch.utils.data import DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size = config["batch_size"],
    shuffle = True,
    num_workers = config["num_workers"],
    pin_memory=True
)

valid_dataloader = DataLoader(
    valid_dataset,
    batch_size = config["batch_size"],
    shuffle = False, 
    num_workers = config["num_workers"],
    pin_memory=True
)

from module.custom_model import Model
model = Model(config["mask_ratio"], config["pretrained"])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

if config["is_import_model"]:
    model.load_state_dict(torch.load(config["model_save_path"] + config["import_model_name"]))

import torch.optim as optim
optimizer = optim.AdamW(model.parameters(),
                        lr=config["lr"],
                        weight_decay=config["weight_decay"])

from module.model_train import model_train
model_train(train_dataloader, model, optimizer, device, config["epochs"], config["model_save_path"])

# from module.calc_score import eval_model, calc_puzzle
# pred_valid_df = eval_model(model, valid_dataloader, valid_df)
# score = calc_puzzle(valid_df, pred_valid_df)
# print("total score:", score)