from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch

class ProconDataset(Dataset):

    def __init__(self, df):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
        """
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        state, score, size = self.data.iloc[idx]
        state = torch.tensor(state, dtype=torch.long)

        max_score = size ** 2 // 2
        score_norm = score / max_score
        
        score_norm = torch.tensor(score_norm, dtype=torch.float32)
        
        
        return state, score_norm
