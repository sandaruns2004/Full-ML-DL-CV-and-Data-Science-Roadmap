import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(PlantDiseaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # 128x128 -> 64x64 -> 32x32 -> 16x16
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train():
    data_dir = '../data/plant_village'
    num_classes = 5
    classes = ['Apple_Healthy', 'Apple_Scab', 'Potato_Healthy', 'Potato_Late_Blight', 'Tomato_Early_Blight']
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if os.path.exists(data_dir):
        print("Loading actual PlantVillage dataset...")
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        classes = dataset.classes
        num_classes = len(classes)
    else:
        print("Dataset not found. Generating synthetic FakeData for demonstration purposes...")
        dataset = datasets.FakeData(size=1000, image_size=(3, 128, 128), num_classes=num_classes, transform=transform)
        
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = PlantDiseaseCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    print("Starting Training Loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(dataloader):.4f}")
        
    os.makedirs('src', exist_ok=True)
    torch.save(model.state_dict(), 'src/model.pth')
    
    # Save classes for inference
    with open('src/classes.json', 'w') as f:
        json.dump(classes, f)
        
    print("Model saved to src/model.pth")

if __name__ == "__main__":
    train()
