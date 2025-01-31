import json
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
from sklearn.model_selection import train_test_split
from env import BATCH_SIZE
from pdb import set_trace
import numpy as np  
def group2eight(input_list , group_size):
    return [input_list[i:i + group_size] for i in range(0, len(input_list), group_size)]


class LargeTextTokenizer:
    """
    Custom dataset to handle large text files for batching.
    """
    def __init__(self, file_path, batch_size , dict_path = './dict.json', ):
        """
        Args:
            file_path (str): Path to the large text file.
            batch_size (int): Number of tokens per batch.
        """
        # self.max_pad_group  = 1002
        self.max_pad_group  = 500
        self.max_pad_group_octa = self.max_pad_group * 8
        self.group_size = 8
        with open(dict_path, 'r') as file:
            self.dict = json.load(file)
        self.file_path = file_path
        self.batch_size = batch_size
        self.tokens = self._load_tokens()
        self.num_batches = len(self.tokens) // batch_size
        print("[+] self.num_batches", self.num_batches)

    def _load_tokens(self):
        """
        Load all tokens from the file.
        Returns:
            List[str]: A list of tokens from the file.
        """
        tokens = []
        with open(self.file_path, "r") as file:
            for line in file:
                song = line.strip().split()
                song.append('</s>')
                song_id = []
                for note in song:
                    song_id.append(self.dict[note])

                #! add pad
                if len(song_id) <  self.max_pad_group_octa :
                    song_id +=  (self.max_pad_group_octa - len(song_id)) * [1]
                elif len(song_id) >  self.max_pad_group_octa:
                    song_id = song_id[:self.max_pad_group_octa]

                song_id_grouped = group2eight(song_id , self.group_size)
                tokens.append(song_id_grouped)
        return np.array(tokens) #torch.tensor(tokens)
    
    def get_tokens(self):
        return self.tokens

class CustomDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens
    
    def __len__(self):
        """
        Returns the total number of batches.
        """
        return len(self.tokens) 

    def mask(self, data):
        mask = (data[:, 4] != 1) & (data[:, 4] != 0) & (data[:, 4] != 2)
        data[mask, 4] = 1236
        return data

    def __getitem__(self, idx):
        """
        Returns a batch of data as a string.
        Args:
            idx (int): Index of the batch.
        Returns:
            str: A string of tokens representing the batch.
        """
        tgt =  self.tokens[idx]
        src = self.mask(tgt.clone())
        return src , tgt

# Custom collate function
def custom_collate(batch):
    """
    Custom collate function to batch src and tgt separately.
    Args:
        batch (list): A list of (src, tgt) pairs.
    Returns:
        tuple: A tuple containing batched src and tgt tensors.
    """
    src_batch = torch.stack([item[0] for item in batch])  # Batch src tensors
    tgt_batch = torch.stack([item[1] for item in batch])  # Batch tgt tensors
    return src_batch, tgt_batch

class DataLoaderMusicBERT:
    def __init__(self , batch_size = 4):
        self.batch_size = batch_size  # Number of songs per batch

    def get_train_data_loader(self):
        return self.train_loader

    def create_train_data_loader(self , file_path = "processed.txt" ,train_input_path = "torch_groove_train.pkl" ,val_input_path = "torch_groove_val.pkl"):
        print("[+] creating tensor with batch of", self.batch_size)

        dataset = LargeTextTokenizer(file_path, self.batch_size)
        # set_trace()
        df_train, df_val = train_test_split(dataset.get_tokens(), test_size=0.2)

        train = CustomDataset(torch.tensor(df_train))
        val = CustomDataset(torch.tensor(df_val))

        self.train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True , pin_memory=True , collate_fn=custom_collate)
        self.val_loader = DataLoader(val, batch_size=self.batch_size, shuffle=True ,pin_memory=True , collate_fn=custom_collate)

        with open(train_input_path, "wb") as f:
            pickle.dump(self.train_loader.dataset, f)

        with open(val_input_path, "wb") as f:
            pickle.dump(self.val_loader.dataset, f)
        
        print("[+] creating tensor Done")

    def load_dataloader(self ,train_input_path = "torch_groove_train.pkl" ,val_input_path = "torch_groove_val.pkl"):
        print("[+] loading tensor")

        with open(train_input_path, "rb") as f:
            dataset = pickle.load(f)        
        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        with open(val_input_path, "rb") as f:
            dataset = pickle.load(f)        
        self.val_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        print("[+] loading tensor Done")

# # Iterate through the dataloader
# for batch_idx, batch in enumerate(train_loader):
#     print(f"Batch {batch_idx}:", batch)  # Access batch[0] as a string
#     # Pass batch[0] (string) to your model for training
#     # Example: output = model(batch[0])
#     break  # Remove this break to process the full dataset
if __name__ == '__main__':
    dl = DataLoaderMusicBERT(batch_size = BATCH_SIZE)
    # dl.load_dataloader()
    dl.create_train_data_loader()
    for batch in dl.get_train_data_loader():
        print(batch)
        break
