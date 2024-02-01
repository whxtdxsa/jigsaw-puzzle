import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

def model_train(train_dataloader, model, optimizer, epochs, save_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    for epoch in range(1, epochs+ 1):
        print('Epoch ', epoch)
        model.train()
        for i, x in tqdm(enumerate(train_dataloader), total=len(train_dataloader),desc=f"epoch {epochs}"):
            x = x.to(device)

            optimizer.zero_grad()
            preds, targets = model(x)
            loss = F.cross_entropy(preds, targets)

            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), save_path + f'/model_state_{epoch}.pth')