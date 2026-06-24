import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_credit_data(num_samples=5000):
    np.random.seed(99)
    
    # Features
    age = np.random.randint(18, 70, size=num_samples)
    income = np.random.randint(20000, 150000, size=num_samples)
    loan_amount = np.random.randint(5000, 50000, size=num_samples)
    credit_history_length = np.random.randint(0, 30, size=num_samples)
    num_existing_loans = np.random.randint(0, 5, size=num_samples)
    
    # Simple risk formulation
    # High risk if loan amount is large relative to income, or short credit history, or many existing loans
    risk_score = (
        (loan_amount / income) * 2.0
        - (credit_history_length * 0.05)
        + (num_existing_loans * 0.4)
        - (age * 0.01)
    )
    
    # Normalize risk score roughly to sigmoid
    risk_prob = 1 / (1 + np.exp(-risk_score))
    default = np.random.binomial(1, risk_prob)
    
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'loan_amount': loan_amount,
        'credit_history_length': credit_history_length,
        'num_existing_loans': num_existing_loans,
        'default': default
    })
    
    return df

class CreditRiskModel(nn.Module):
    def __init__(self, input_dim):
        super(CreditRiskModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return self.sigmoid(x)

def train():
    data_dir = '../data'
    os.makedirs(data_dir, exist_ok=True)
    
    print("Generating Synthetic Credit Risk Data...")
    df = generate_credit_data()
    df.to_csv(f"{data_dir}/credit_risk_data.csv", index=False)
    
    X = df.drop('default', axis=1).values
    y = df['default'].values.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')
    
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    
    model = CreditRiskModel(input_dim=X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 150
    print("Training model...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 30 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to model.pth")

if __name__ == "__main__":
    train()
