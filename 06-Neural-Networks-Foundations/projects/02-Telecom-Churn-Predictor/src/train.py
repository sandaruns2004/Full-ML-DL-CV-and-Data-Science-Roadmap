import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_synthetic_data(num_samples=5000):
    np.random.seed(42)
    
    # Features
    tenure = np.random.randint(1, 72, size=num_samples)
    monthly_charges = np.random.uniform(20, 120, size=num_samples)
    total_charges = tenure * monthly_charges + np.random.normal(0, 50, size=num_samples)
    tech_support = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])
    contract_type = np.random.choice([0, 1, 2], size=num_samples, p=[0.5, 0.3, 0.2]) # 0: Month, 1: 1yr, 2: 2yr
    
    # Calculate churn probability roughly based on features
    churn_prob = (
        0.5 
        - 0.005 * tenure 
        + 0.003 * monthly_charges 
        - 0.2 * tech_support 
        - 0.15 * contract_type
    )
    # Sigmoid function equivalent to get probability between 0 and 1
    churn_prob = 1 / (1 + np.exp(-churn_prob))
    
    # Generate labels
    churn = np.random.binomial(1, churn_prob)
    
    df = pd.DataFrame({
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'tech_support': tech_support,
        'contract_type': contract_type,
        'churn': churn
    })
    
    return df

class ChurnPredictor(nn.Module):
    def __init__(self, input_dim):
        super(ChurnPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return self.sigmoid(x)

def train():
    data_dir = '../data'
    os.makedirs(data_dir, exist_ok=True)
    
    print("Generating synthetic Telco Churn dataset...")
    df = generate_synthetic_data()
    df.to_csv(f"{data_dir}/telco_churn_synthetic.csv", index=False)
    
    X = df.drop('churn', axis=1).values
    y = df['churn'].values.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for inference in app.py
    joblib.dump(scaler, 'scaler.pkl')
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    
    model = ChurnPredictor(input_dim=X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 100
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    print("Training complete. Saving model weights...")
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to model.pth")

if __name__ == "__main__":
    train()
