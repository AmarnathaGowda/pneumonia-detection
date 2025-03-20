import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.architecture import PneumoniaClassifier
from data.dataset import get_data_loaders
from utils.config import DEVICE, NUM_CLASSES
from utils.hyperparams import LEARNING_RATES, BATCH_SIZES, NUM_EPOCHS_LIST
from utils.config import DATA_PATH

# Load data
data_dir = DATA_PATH
train_loader, val_loader, _ = get_data_loaders(data_dir, batch_size=32)  # Temporary batch size

# Define cross-validation
k_folds = 3
kf = KFold(n_splits=k_folds, shuffle=True)

best_accuracy = 0
best_params = {}

for lr in LEARNING_RATES:
    for batch_size in BATCH_SIZES:
        for num_epochs in NUM_EPOCHS_LIST:
            print(f"Testing LR: {lr}, Batch Size: {batch_size}, Epochs: {num_epochs}")
            
            fold_accuracies = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(train_loader.dataset)):
                print(f"Fold {fold+1}/{k_folds}")
                
                # Create data loaders for this fold
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
                val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
                train_loader_fold = torch.utils.data.DataLoader(
                    train_loader.dataset, batch_size=batch_size, sampler=train_subsampler)
                val_loader_fold = torch.utils.data.DataLoader(
                    train_loader.dataset, batch_size=batch_size, sampler=val_subsampler)
                
                # Initialize model
                model = PneumoniaClassifier(num_classes=NUM_CLASSES).to(DEVICE)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                # Train the model
                for epoch in range(num_epochs):
                    model.train()
                    for images, labels in train_loader_fold:
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                
                # Evaluate on validation fold
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in val_loader_fold:
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        outputs = model(images)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                accuracy = correct / total
                fold_accuracies.append(accuracy)
            
            avg_accuracy = sum(fold_accuracies) / k_folds
            print(f"Average Accuracy: {avg_accuracy:.4f}")
            
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_params = {'lr': lr, 'batch_size': batch_size, 'num_epochs': num_epochs}

print(f"Best Parameters: {best_params}")
print(f"Best Accuracy: {best_accuracy:.4f}")


# Retrain with best parameters
best_lr = best_params['lr']
best_batch_size = best_params['batch_size']
best_num_epochs = best_params['num_epochs']

# Update data loaders with best batch size
train_loader, val_loader, _ = get_data_loaders(data_dir, batch_size=best_batch_size)

# Initialize and train the model
model = PneumoniaClassifier(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=best_lr)

for epoch in range(best_num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Save the optimized model
torch.save(model.state_dict(), 'model/optimized_pneumonia_classifier.pth')
print("Optimized model saved successfully.")