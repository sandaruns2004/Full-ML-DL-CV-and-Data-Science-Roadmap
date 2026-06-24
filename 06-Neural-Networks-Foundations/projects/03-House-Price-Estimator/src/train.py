import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HousePriceNN(nn.Module):
    def __init__(self, input_dim):
        super(HousePriceNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def train():
    print("Generating synthetic House Price dataset...")
    np.random.seed(42)
    n = 1000
    sqft = np.random.uniform(800, 5000, n)
    bedrooms = np.random.randint(1, 6, n)
    age = np.random.uniform(0, 100, n)
    
    price = (sqft * 150) + (bedrooms * 25000) - (age * 1000) + np.random.normal(0, 20000, n)
    
    data = pd.DataFrame({'sqft': sqft, 'bedrooms': bedrooms, 'age': age, 'price': price})
    
    X = data[['sqft', 'bedrooms', 'age']].values
    y = data['price'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    model = HousePriceNN(input_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training model...")
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        
    print(f"Final Training Loss (MSE): {loss.item():.4f}")
    
    os.makedirs('src', exist_ok=True)
    torch.save(model.state_dict(), 'src/model.pth')
    print("Model saved to src/model.pth")

if __name__ == "__main__":
    train()
