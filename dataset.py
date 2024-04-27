from torch.utils.data import Dataset, DataLoader
import torch
class CausalLMDataset(Dataset):
    def __init__(self,text,block_size):
        self.text = text
        self.block_size = block_size

    def __len__(self):
        return max(len(self.text) - self.block_size,1)

    def __getitem__(self, idx):
        "perform input shifting to get labels"
        return (
            torch.tensor(self.text[idx:idx+self.block_size],dtype=torch.long),
            torch.tensor(self.text[idx+1:idx+self.block_size+1],dtype=torch.long),
        )