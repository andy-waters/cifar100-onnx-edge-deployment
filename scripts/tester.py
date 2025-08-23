"""
tester.py

Provides model evaluation utilities for CIFAR-100 experiments.
"""
import warnings
warnings.filterwarnings("ignore") 

import torch
from torch import nn, optim
from tqdm import tqdm  # For progress bar
from torchvision import datasets, transforms
from resnet_model import ResNetModelBuilder
from resnet_model import ResNetModelType
from cifar100_data_provider import CIFAR100DataProvider
from cifar100_data_provider import DatasetType
import time


class Tester:
    """
    Utility class for testing models on CIFAR-100.
    """
    def __init__(self):
        pass

    @staticmethod
    def test_model(test_loader, model, criterion, device):
        """
        Evaluates a model on the test set.
        Args:
            test_loader (DataLoader): Test data loader.
            model (torch.nn.Module): Model to evaluate.
            criterion: Loss function.
            device: Device to run evaluation on.
        Returns:
            tuple: (test_loss, accuracy, total_time)
        """
        # ...existing code...
        # Set the model to evaluation mode
        model.eval()
        
        test_loss = 0
        correct = 0
        total = 0
        total_time = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                start_time = time.time()
                outputs = model(images)
                end_time = time.time()
                total_time += end_time - start_time
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                # Get predictions and count correct predictions
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        # Calculate average loss and accuracy
        test_loss /= len(test_loader)
        test_accuracy = correct / total
        avg_inf_time = (total_time / len(test_loader) * 1000)
        
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print(f"Avg. Inference time: {avg_inf_time:.2f} ms")

if __name__ == '__main__':
    # Example usage
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the CIFAR-100 data provider
    data_provider = CIFAR100DataProvider(batch_size=32)
    # Load the test data
    _, _, test_loader = data_provider.get_data_loaders(dataset_type=DatasetType.TEST)

    # Load the trained model
    model = ResNetModelBuilder.get_model(model_type=ResNetModelType.RESNET34, num_classes=100, pretrained=True)
    model.load_state_dict(torch.load('output/resnet_best_model.pth', map_location=device))
    model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Test the model
    Tester.test_model(test_loader, model, criterion, device)
  
