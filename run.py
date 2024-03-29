from module.data_download import download_and_unzip, download_lib
download_lib()

import yaml
with open('./config_run.yaml') as f:
    config = yaml.safe_load(f)

from module.data_loader import get_dataframe, get_dataset_test
train_df, valid_df, test_df = get_dataframe(config["data_path"], config["train_size"], config["test_size"])

test_dataset = get_dataset_test(test_df, config["data_path"])

import torch
from torch.utils.data import DataLoader

test_dataloader = DataLoader(
    test_dataset,
    batch_size = config["batch_size"],
    shuffle = False, 
    num_workers = config["num_workers"],
    pin_memory=True
)

from module.custom_model import Model
model = Model(config["mask_ratio"], config["pretrained"], config["num_fc_layer"])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

model.load_state_dict(torch.load(config["model_save_path"] + str(config["num_fc_layer"]) + config["import_model_name"]))

from module.model_eval import eval_model
test_pred_df = eval_model(model, test_dataloader, test_df)
test_pred_df.to_csv(config["result_save_path"] + str(config["num_fc_layer"]) + config["import_model_name"] + ".csv", index=False)