import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from data_loader import DataLoaderMusicBERT
from pdb import set_trace
from time import sleep
from copy import deepcopy
class StandaloneMusicBERTModel(torch.nn.Module):
    def __init__(self, model, dictionary):
        super().__init__()
        self.model = model
        self.dictionary = dictionary

    def token2id(self,batch):
        """Convert grouped input tokens to tensor IDs and pass through the model.
        Batch x song x group x tokens
        """
        batched_song = []
        song_lens = []
        for song in batch:
            song_ids = []
            song_lens.append(len(song))
            for token_group in song:
                if len(token_group) != 8:
                    raise ValueError("Each token group must contain exactly 8 tokens.")
                group_ids = [self.dictionary.index(token) for token in token_group]
                song_ids.append(group_ids)
            batched_song.append(song_ids)
        max_len = max(song_lens)
        for c , song in enumerate(batched_song):
            required_pad = max_len - len(song)
            song.extend([[1 for _ in range(8)] for _ in range(required_pad)]) # 1 is for pad
            batched_song[c] = song

        return batched_song

    def forward(self, grouped_input_tokens):
        batched_id = self.token2id(grouped_input_tokens)
        # set_trace()
        input_tensor = torch.tensor(batched_id)# .unsqueeze(0)  # Add batch dimension
        # return self.model(**{"src_tokens": input_tensor})
        # Flatten the sequence and group dimensions
        batch_size, sequence_length, group_length = input_tensor.shape
        input_tensor = input_tensor.view(batch_size, -1)  # Shape: [batch_size, sequence_length * group_length]

        # Pass to the model
        return self.model(**{"src_tokens": input_tensor})

# # Dummy dataset for autoencoding
# class AutoencodingDataset(Dataset):
#     def __init__(self, data):
#         """
#         data: list of tokenized sequences (e.g., MIDI sequences)
#         """
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         input_seq = self.data[idx]
#         # Target is the same as input
#         target_seq = input_seq.clone()
#         return input_seq, target_seq

# Define the training function
def train_autoencoder(model, num_epochs=50, batch_size=16, lr=1e-4, device="cuda"):
    """
    model: Pre-trained MusicBERT model
    dataset: Dataset containing input-output pairs
    num_epochs: Number of training epochs
    batch_size: Training batch size
    lr: Learning rate for the optimizer
    device: "cuda" or "cpu"
    """
    # Move model to device
    model.to(device)
    model.train()

    dl = DataLoaderMusicBERT()
    dl_train = dl.get_train_data_loader()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    pad_index = 1
    criterion = nn.CrossEntropyLoss(ignore_index = pad_index)  # Handle padding index

    total_batches = len(dl_train)  # Total number of batches in DataLoader    
    for epoch in range(num_epochs):
        epoch_desc = f"[++] Epoch {epoch + 1}/{num_epochs} (0.00%)"
        with tqdm(total=total_batches, desc=epoch_desc, unit="batch") as batch_pbar:
            for batch_idx, batch in enumerate(dl_train):
                epoch_percentage = ((epoch * total_batches + batch_idx + 1) / (num_epochs * total_batches)) * 100
                # set_trace()
                batch = batch[0] #TODO repeated dim. should be because of dataloader implementation. each batch should be group of songs

                batch_pbar.set_description(f"[+] Epoch {epoch + 1}/{num_epochs} ({epoch_percentage:.2f}%)")
                batch_pbar.update(1)
                batch_pbar.refresh()
                # Process the batch
                output = model(batch)[0] 
                output_flat = output.view(output.shape[0]*output.shape[1] , -1)

                target = deepcopy(batch)

                target_id = model.token2id(target)
                # set_trace()
                target_tensor = torch.tensor(target_id)# .unsqueeze(0)  # Add batch dimension
                # return self.model(**{"src_tokens": input_tensor})
                # Flatten the sequence and group dimensions
                targe_batch_size, targe_sequence_length, targe_group_length = target_tensor.shape
                target_tensor_flat = target_tensor.view(targe_batch_size * targe_sequence_length * targe_group_length)  # Shape: [batch_size, sequence_length * group_length]


                loss = criterion(output_flat, target_tensor_flat)

                # epoch_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_pbar.set_postfix(loss=loss.item())
            
    print("Training Complete!")

def test(model):
    model_path = "standalone_musicbert_model.pth"
    model = torch.load(model_path)
    model.eval()
    # example_grouped_tokens = [
    #     ["<s>", "<mask>", "<0-0>", "<1-0>", "<2-0>", "<3-0>", "<4-0>", "<5-0>"],
    #     ["<6-0>", "<7-0>", "<mask>", "<0-1>", "<1-1>", "<2-1>", "<3-1>", "</s>"]
    # ]
    example_grouped_tokens = [
        ["<s>", "<mask>", "<0-0>", "<1-0>", "<2-0>", "<3-0>", "<4-0>", "<5-0>" , "<6-0>", "<7-0>", "<mask>", "<0-1>", "<1-1>", "<2-1>", "<3-1>", "</s>"]
    ]

    output = model(example_grouped_tokens)
    print("Output:", output , output[0].shape)

if __name__ == '__main__':
    model_path = "standalone_musicbert_model.pth"
    model = torch.load(model_path)
    # dataset = AutoencodingDataset(dummy_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] device {device}")
    train_autoencoder(model, batch_size=16, lr=1e-4, device=device)
