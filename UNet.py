import os
import torch
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import segmentation_models_pytorch as smp

# Prepare the Dataset
# Create a custom dataset class to handle loading and transforming images
 
class DefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        for label, class_dir in enumerate(['no_defect', 'defect']):
            class_path = os.path.join(root_dir, class_dir)
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                self.data.append(file_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('L')  # Grayscale
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


# Training and validation image transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# Training pipeline
# Define paths
train_dir = './images/train'
val_dir = './images/val'

# Create datasets and data loaders
train_dataset = DefectDataset(train_dir, transform=transform)
val_dataset = DefectDataset(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Model
class UNetClassifier(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=1):
        super(UNetClassifier, self).__init__()
        self.base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1  # Placeholder for the encoder
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Reduce spatial dimensions to 1x1
            nn.Flatten(),                 # Convert to shape [batch_size, 512]
            nn.Linear(512, 1),            # Fully connected layer to reduce to a single value
            nn.Sigmoid()                  # Probability output
        )

    def forward(self, x):
        # Extract features from encoder
        encoder_output = self.base_model.encoder(x)
        last_feature_map = encoder_output[-1]  # Use the deepest feature map: [batch_size, 512, 4, 4]

        # Apply global pooling and fully connected layer
        x = self.global_pool(last_feature_map)  # Output shape: [batch_size, 1]
        return x

# Initialize model, loss function, and optimizer
model = UNetClassifier()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images, labels.unsqueeze(1)  # Shape [batch_size, 1]
        outputs = model(images)  # Shape [batch_size, 1]
        loss = criterion(outputs, labels)

        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation Loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images, labels.unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss/len(train_loader):.4f}, "
          f"Val Loss: {val_loss/len(val_loader):.4f}")


# Model saving
torch.save(model.state_dict(), "defect_model.pth")

# Model testing
# Load model
model.load_state_dict(torch.load("defect_model.pth"))
model.eval()

# Test on a new image
test_image_path = './images/test/frame_482.jpg'
test_image = Image.open(test_image_path).convert('L')
test_image = transform(test_image).unsqueeze(0)

with torch.no_grad():
    probability = model(test_image).item()
    print(f"Probability of defect: {probability:.4f}")
    if probability > 0.35:
        print("Defect detected.")
    else:
        print("No defect detected.")
