import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import numpy as np
import cv2
from model import create_model

class GameStateDataset(Dataset):
   def __init__(self, data_dir, transform=None):
       self.data_dir = Path(data_dir)
       self.transform = transform
       self.classes = ['menu', 'gameplay', 'loading']
       self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
       
       self.images = []
       self.labels = []
       
       print("\nloading dataset")
       for class_name in self.classes:
           class_dir = self.data_dir / class_name
           if not class_dir.exists():
               print(f"warning: directory {class_name} not found")
               continue
               
           class_images = list(class_dir.glob('*.png'))
           print(f"Found {len(class_images)} images in {class_name}")
           
           for img_path in class_images:
               self.images.append(str(img_path))
               self.labels.append(self.class_to_idx[class_name])
       
       print(f"total images loaded: {len(self.images)}")

   def __len__(self):
       return len(self.images)

   def __getitem__(self, idx):
       try:
           img_path = self.images[idx]
           image = cv2.imread(img_path)
           if image is None:
               raise ValueError(f"failed to load image: {img_path}")
           
           image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           
           if self.transform:
               image = self.transform(image)
               
           label = self.labels[idx]
           return image, label
       except Exception as e:
           print(f"error loading image {idx}: {str(e)}")
           raise

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
   model = model.to(device)
   best_val_acc = 0.0
   
   for epoch in range(num_epochs):
       print(f'\nEpoch {epoch+1}/{num_epochs}')
       print('-' * 10)
       
       model.train()
       running_loss = 0.0
       running_corrects = 0
       total_batches = len(train_loader)
       
       print(f"processing {total_batches} training batches...")
       
       for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
           try:
               inputs = inputs.to(device)
               labels = labels.to(device)
               
               optimizer.zero_grad()
               
               outputs = model(inputs)
               _, preds = torch.max(outputs, 1)
               loss = criterion(outputs, labels)
               
               loss.backward()
               optimizer.step()
               
               running_loss += loss.item() * inputs.size(0)
               running_corrects += torch.sum(preds == labels.data)
               
               if batch_idx % 5 == 0:
                   print(f"batch {batch_idx}/{total_batches}")
                   
           except Exception as e:
               print(f"error in batch {batch_idx}: {str(e)}")
               continue
       
       epoch_loss = running_loss / len(train_loader.dataset)
       epoch_acc = running_corrects.double() / len(train_loader.dataset)
       
       print(f'train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
       
       model.eval()
       running_loss = 0.0
       running_corrects = 0
       
       print("starting validation...")
       with torch.no_grad():
           for inputs, labels in val_loader:
               inputs = inputs.to(device)
               labels = labels.to(device)
               
               outputs = model(inputs)
               _, preds = torch.max(outputs, 1)
               loss = criterion(outputs, labels)
               
               running_loss += loss.item() * inputs.size(0)
               running_corrects += torch.sum(preds == labels.data)
       
       val_loss = running_loss / len(val_loader.dataset)
       val_acc = running_corrects.double() / len(val_loader.dataset)
       
       print(f'val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
       
       if val_acc > best_val_acc:
           best_val_acc = val_acc
           torch.save(model.state_dict(), 'l4d2_model.pth')
           print("saved new best model!")
   
   return model

def main():
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f'using device: {device}')
   
   transform = transforms.Compose([
       transforms.ToPILImage(),
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   
   try:
       print("creating dataset...")
       full_dataset = GameStateDataset('training_data', transform=transform)
       
       if len(full_dataset) == 0:
           raise ValueError("no images found in the dataset")
       
       train_size = int(0.8 * len(full_dataset))
       val_size = len(full_dataset) - train_size
       train_dataset, val_dataset = torch.utils.data.random_split(
           full_dataset, [train_size, val_size]
       )
       
       print(f"training set size: {len(train_dataset)}")
       print(f"validation set size: {len(val_dataset)}")
       
       train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
       val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
       
       print("creating model...")
       model = create_model(num_classes=3)
       criterion = nn.CrossEntropyLoss()
       optimizer = optim.Adam(model.parameters(), lr=0.001)
       
       print("starting training...")
       model = train_model(
           model=model,
           train_loader=train_loader,
           val_loader=val_loader,
           criterion=criterion,
           optimizer=optimizer,
           num_epochs=10,
           device=device
       )
       
   except Exception as e:
       print(f"an error occurred: {str(e)}")
       import traceback
       traceback.print_exc()

if __name__ == '__main__':
   main()