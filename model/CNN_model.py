import torch
import torch.nn as nn
import torch.nn.functional as F 

class CNN_Network(nn.Module):
    def __init__(self, n_values, n_hidden = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = n_values, embedding_dim = 8)
        
        self.conv = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )
    
    def forward(self, x):
        x = self.embedding(x.long())
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)