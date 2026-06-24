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

def generate_hr_data(num_samples=4000):
    np.random.seed(123)
    
    # Features
    satisfaction = np.random.uniform(0.1, 1.0, size=num_samples)
    last_evaluation = np.random.uniform(0.3, 1.0, size=num_samples)
    number_project = np.random.randint(2, 8, size=num_samples)
    average_monthly_hours = np.random.randint(96, 310, size=num_samples)
    time_spend_company = np.random.randint(2, 11, size=num_samples)
    work_accident = np.random.choice([0, 1], size=num_samples, p=[0.85, 0.15])
    promotion_last_5years = np.random.choice([0, 1], size=num_samples, p=[0.95, 0.05])
    
    # Calculate flight risk based on specific conditions (e.g., overworked, low satisfaction)
    attrition_score = (
        -2.0 * satisfaction 
        + 0.5 * (average_monthly_hours / 200) 
        + 0.3 * (time_spend_company / 5)
        - 1.0 * promotion_last_5years
    )
    
    attrition_prob = 1 / (1 + np.exp(-attrition_score))
    attrition = np.random.binomial(1, attrition_prob)
    
    df = pd.DataFrame({
        'satisfaction_level': satisfaction,
        'last_evaluation': last_evaluation,
        'number_project': number_project,
        'average_montly_hours': average_monthly_hours,
        'time_spend_company': time_spend_company,
        'Work_accident': work_accident,
        'promotion_last_5years': promotion_last_5years,
        'left': attrition
    })
    
    return df

class AttritionModel(nn.Module):
    def __init__(self, input_dim):
        super(AttritionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return self.sigmoid(x)

def train():
    data_dir = '../data'
    os.makedirs(data_dir, exist_ok=True)
    
    print("Generating Synthetic HR Attrition Data...")
    df = generate_hr_data()
    df.to_csv(f"{data_dir}/hr_attrition_data.csv", index=False)
    
    X = df.drop('left', axis=1).values
    y = df['left'].values.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')
    
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    
    model = AttritionModel(input_dim=X.shape[1]).to(device)
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
