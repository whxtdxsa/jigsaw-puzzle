import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

def model_train(train_dataloader, model, optimizer, device, epochs, save_path):
    for epoch in range(1, epochs+ 1):
        model.train()
        for i, x in tqdm(enumerate(train_dataloader), total=len(train_dataloader),desc=f"epoch {epoch}"):
            x = x.to(device)

            optimizer.zero_grad()
            preds, targets = model(x)
            loss = F.cross_entropy(preds, targets)

            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), save_path + f'/model_state_{epoch}.pth')