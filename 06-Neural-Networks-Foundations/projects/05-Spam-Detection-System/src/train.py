import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_text_data():
    # Synthetic spam and ham messages for demonstration
    ham_messages = [
        "Hey, are we still on for lunch today?",
        "Don't forget to submit your report by 5 PM.",
        "Can you pick up some milk on the way home?",
        "See you at the meeting tomorrow.",
        "Thanks for the update!",
        "I'll call you back later.",
        "What time is the movie starting?",
        "Please review the attached document.",
        "Happy birthday! Have a great day.",
        "Let me know if you need any help with that."
    ] * 200 # Duplicate to create a dataset
    
    spam_messages = [
        "CONGRATULATIONS! You've won a $1000 Walmart gift card. Click here to claim.",
        "URGENT: Your bank account has been locked. Verify your identity at this link.",
        "You have been selected for a free vacation to the Bahamas! Call now.",
        "Get rich quick! Invest in this new cryptocurrency and double your money.",
        "Limited time offer: Get 80% off all designer watches.",
        "Your package is waiting for delivery. Pay the $2 shipping fee here.",
        "Meet hot singles in your area tonight!",
        "Claim your tax refund of $500 immediately by submitting this form.",
        "Warning: Your PC is infected with a virus. Download our antivirus software now.",
        "You've been pre-approved for a $50,000 loan. Click to apply."
    ] * 200

    messages = ham_messages + spam_messages
    labels = [0] * len(ham_messages) + [1] * len(spam_messages)
    
    df = pd.DataFrame({'text': messages, 'label': labels})
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

class SpamClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SpamClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

def train():
    data_dir = '../data'
    os.makedirs(data_dir, exist_ok=True)
    
    print("Generating Synthetic Spam/Ham dataset...")
    df = get_text_data()
    df.to_csv(f"{data_dir}/spam_data.csv", index=False)
    
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()
    
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    X_train_t = torch.FloatTensor(X_train_vec).to(device)
    y_train_t = torch.FloatTensor(y_train.values).reshape(-1, 1).to(device)
    
    model = SpamClassifier(input_dim=X_train_vec.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 50
    print("Training model...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to model.pth")

if __name__ == "__main__":
    train()
