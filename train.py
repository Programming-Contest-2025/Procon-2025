import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
from data.dataLoader import ProconDataset
import torch.nn as nn 
import torch.optim as optim
from model.CNN_model import CNN_Network

def train_cnn_model(train_loader, val_loader, size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_values = size * size // 2
    cnn_model = CNN_Network(n_values=n_values)
    cnn_model = cnn_model.to(device=device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr = 1e-3)
    
    num_epochs = size * 5
    
    for epoch in range(num_epochs):
        cnn_model.train()
        running_loss = 0.0
        
        for states, target_scores in train_loader:
            states, target_scores = states.to(device), target_scores.to(device)
            
            optimizer.zero_grad()
            outputs = cnn_model(states).squeeze()
            loss = criterion(outputs, target_scores.float())
            loss.backward()
            optimizer.step() 
            
            running_loss += loss
        avg_train_loss = running_loss / len(train_loader)
        
        cnn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for states, target_scores in val_loader:
                states, target_scores = states.to(device), target_scores.to(device)
                outputs = cnn_model(states).squeeze()
                loss = criterion(outputs, target_scores.float())
                val_loss += val_loss
            avg_val_loss = val_loss / len(val_loader)
            
            if epoch % 5 == 0:
                print(f"Size = {size} | Epoch = [{epoch}/{num_epochs}] | Train_loss: {avg_train_loss:.4f} | Val_loss: {avg_val_loss:.4f}")
        torch.save(cnn_model.state_dict(), f"model/weights/cnn_model_{size}_weight.pth")
        print(f"✅ Model saved to cnn_model_{size}.pth")

def train(df):
    for size in range(4, 25, 2):
        df = ProconDataset(df[df['Size'] == size])
        
        #Tạo train_set để huấn luyện, test để đánh giá
        total = len(df)
        train_size = int(0.6 * total)
        val_size = int(0.2 * total)
        test_size = total - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(df, 
                            [train_size, val_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size = 16, shuffle = False)
        test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = False)

        train_cnn_model(train_loader, val_loader, size)
        print("*" * 20)

if __name__ == "__main__":
    df = pd.read_csv("data/trainSet_processed.csv")
    train(df=df)