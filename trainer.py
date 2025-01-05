import torch
from torch import nn, optim
from tqdm import tqdm
from data_loader import DataLoaderMusicBERT, CustomDataset
# from pdb import set_trace
from time import time,ctime
from copy import deepcopy
from statistics import mean
import wandb
import os 
from env import wandbAPI
EPOCHS = 1000
SAVE_INTERVAL = 100
CODE_RUN_TIME = ctime(time()).replace(':','_').replace(' ','_')
STATE_SAVE_PATH = os.path.join('data','state',CODE_RUN_TIME)

class StandaloneMusicBERTModel(torch.nn.Module):
    def __init__(self, model, dictionary):
        super().__init__()
        self.model = model
        self.dictionary = dictionary

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        input_tensor = input_tensor.view(batch_size, -1)  # Shape: [batch_size, sequence_length * group_length]
        return self.model(**{"src_tokens": input_tensor})

def train_autoencoder(model, batch_size=16, lr=1e-4, device="cuda"):
    wandb.login(key=wandbAPI)
    wandb.init(project="MusicBERT" )
    wandb.watch(model)

    model.to(device)
    dl = DataLoaderMusicBERT(batch_size=1)
    dl.load_dataloader()
    dl_train = dl.train_loader
    dl_val = dl.val_loader

    optimizer = optim.Adam(model.parameters(), lr=lr)
    pad_index = 1
    criterion = nn.CrossEntropyLoss() 

    total_batches = len(dl_train)  # Total number of batches in DataLoader    
    for epoch in range(EPOCHS):
        model.train()
        epoch_desc = f"[++] Epoch {epoch + 1}/{EPOCHS} (0.00%)"
        losses = []
        # set_trace()
        with tqdm(total=total_batches, desc=epoch_desc, unit="batch") as batch_pbar:
            for batch_idx, batch in enumerate(dl_train):
                epoch_percentage = ((epoch * total_batches + batch_idx + 1) / (EPOCHS * total_batches)) * 100
                batch_pbar.set_description(f"[+] Epoch {epoch + 1}/{EPOCHS} ({epoch_percentage:.2f}%)")
                batch_pbar.update(1)
                batch_pbar.refresh()
                # Process the batch
                src =batch[0].to(device)
                output = model(src)[0] 
                output_flat = output.view(output.shape[0]*output.shape[1] , -1) #! last dim is prob accross all vocabs
                tgt = batch[1].to(device)

                ####
                # Flatten target tensor
                tgt_flat = tgt.view(-1)

                # Create mask for the fourth tokens in each group
                group_len = src.shape[-1]
                # set_trace()
                fourth_mask = torch.arange(group_len, device=device) % group_len == 4  # Mask for 4th token
                fourth_mask = fourth_mask.expand(src.shape).reshape(-1)  # Apply across batch and sequence

                # Exclude pad, <s>, and </s> tokens
                valid_mask = (tgt_flat != pad_index) & (tgt_flat != model.dictionary.index("<s>")) & (tgt_flat != model.dictionary.index("</s>"))
                final_mask = fourth_mask & valid_mask

                # Filter logits and target tokens using the mask
                masked_logits = output_flat[final_mask]  # Logits for masked tokens
                masked_targets = tgt_flat[final_mask]  # Targets for masked tokens

                # Compute loss
                loss = criterion(masked_logits, masked_targets)
                losses.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward() #! time consuming
                optimizer.step()
                
                batch_pbar.set_postfix(loss=loss.item())

        wandb.log({"trainLoss": float(mean(losses))},step=epoch)
        batch_pbar.set_postfix(loss=mean(losses))


        if epoch % SAVE_INTERVAL == 0:
            print("[+] validating")
            model = model.eval()
            losses = []
            for batch_idx, batch in enumerate(dl_val):
                src =batch[0].to(device)
                output = model(src)[0] 
                output_flat = output.view(output.shape[0]*output.shape[1] , -1) #! last dim is prob accross all vocabs
                tgt = batch[1].to(device)
                tgt_flat = tgt.view(-1)
                group_len = src.shape[-1]
                fourth_mask = torch.arange(group_len, device=device) % group_len == 4  # Mask for 4th token
                fourth_mask = fourth_mask.expand(src.shape).reshape(-1)  # Apply across batch and sequence
                valid_mask = (tgt_flat != pad_index) & (tgt_flat != model.dictionary.index("<s>")) & (tgt_flat != model.dictionary.index("</s>"))
                final_mask = fourth_mask & valid_mask
                masked_logits = output_flat[final_mask]  # Logits for masked tokens
                masked_targets = tgt_flat[final_mask]  # Targets for masked tokens
                loss = criterion(masked_logits, masked_targets)
                losses.append(loss.item())

            wandb.log({"valLoss": float(mean(losses))},step=epoch)
            print(f"[+] validation loss {mean(losses)}")

            
            
    print("Training Complete!")

if __name__ == '__main__':
    model_path = "standalone_musicbert_model.pth"
    model = torch.load(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] device {device}")
    train_autoencoder(model, batch_size=16, lr=1e-4, device=device)
