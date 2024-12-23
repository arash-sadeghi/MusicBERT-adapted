import torch
from torch.utils.data import Dataset, DataLoader

def group2eight(input_list , group_size=8):
    return [input_list[i:i + group_size] for i in range(0, len(input_list), group_size)]


class LargeTextDataset(Dataset):
    """
    Custom dataset to handle large text files for batching.
    """
    def __init__(self, file_path, batch_size):
        """
        Args:
            file_path (str): Path to the large text file.
            batch_size (int): Number of tokens per batch.
        """
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
                song = group2eight(song)
                tokens.append(song)
        return tokens

    def __len__(self):
        """
        Returns the total number of batches.
        """
        return self.num_batches

    def __getitem__(self, idx):
        """
        Returns a batch of data as a string.
        Args:
            idx (int): Index of the batch.
        Returns:
            str: A string of tokens representing the batch.
        """
        start = idx * self.batch_size
        end = start + self.batch_size
        return self.tokens[start:end]

# Custom collate function to preserve lists
def custom_collate_fn(batch):
    """
    Custom collate function to avoid conversion to tuples or tensors.
    Args:
        batch (list): The batch to process (list of lists).
    Returns:
        list: The batch as-is.
    """
    return batch  # Return the batch as-is without modification

# Create the dataset and dataloader
def create_dataloader(file_path, batch_size):
    dataset = LargeTextDataset(file_path, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False , collate_fn=custom_collate_fn)
    return dataloader

class DataLoaderMusicBERT:
    def __init__(self):
        file_path = "example_training_data.txt"  # Path to the 6GB file
        batch_size = 4  # Number of tokens per batch
        self.train_loader = create_dataloader(file_path, batch_size)
    def get_train_data_loader(self):
        return self.train_loader

# # Iterate through the dataloader
# for batch_idx, batch in enumerate(train_loader):
#     print(f"Batch {batch_idx}:", batch)  # Access batch[0] as a string
#     # Pass batch[0] (string) to your model for training
#     # Example: output = model(batch[0])
#     break  # Remove this break to process the full dataset
