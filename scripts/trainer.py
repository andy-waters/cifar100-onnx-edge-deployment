"""
trainer.py

Provides training utilities, including early stopping, for CIFAR-100 experiments.
"""
import warnings
warnings.filterwarnings("ignore")  # Ignore warnings

import torch
from torch import nn, optim
from tqdm import tqdm  # For progress bar
from cifar100_data_provider import CIFAR100DataProvider
from cifar100_data_provider import DatasetType
from resnet_model import ResNetModelBuilder
from resnet_model import ResNetModelType
import numpy as np


# Early stopping class (reuse the implementation shared earlier)
class EarlyStopping:
    """
    Implements early stopping to terminate training when validation loss stops improving.
    """
    def __init__(self, patience=10, delta=0.01, path='output/resnet_best_model.pth', verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}). Saving model...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Update the Trainer class
class Trainer:
    def __init__(self):
        pass

    def train_and_validate(train_loader, val_loader, model, criterion, optimizer, device, epochs):
        early_stopping = EarlyStopping(patience=10, verbose=True)  # Initialize EarlyStopping
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Training phase
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            for images, labels in tqdm(train_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
            
            train_loss /= len(train_loader)
            train_accuracy = correct / total
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            # Validation phase
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader):
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            
            # Check early stopping
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

# Python
if __name__ == '__main__':
    #create the device using MPS and/or GPU, but if not available, use CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the CIFAR-100 data provider
    data_provider = CIFAR100DataProvider(batch_size=32)
    #load the validation and training data
    train_loader, val_loader, test_loader = data_provider.get_data_loaders(dataset_type=DatasetType.TRAIN_VAL)


    # Define hyperparameters
    learning_rate = 0.01
    epochs = 5


    #create the resnet model of choice
    resnet = ResNetModelBuilder.get_model(model_type=ResNetModelType.RESNET34, num_classes=100, pretrained=True)

    resnet.to(device)
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=learning_rate)
    # Move the model to the device
    
    
    Trainer.train_and_validate(train_loader, val_loader, resnet, criterion, optimizer, device, epochs)