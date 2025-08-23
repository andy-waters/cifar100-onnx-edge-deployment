"""
cifar100_data_provider.py

Provides data loading utilities for CIFAR-100, including train/val/test splits and model-specific transforms.
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from enum import Enum
from sklearn.model_selection import train_test_split
from resnet_model import ResNetModelType
from torchvision import models

class DatasetType(Enum):
    ALL = "all"
    TRAIN_VAL = "train_val"
    TEST = "test"



class CIFAR100DataProvider:

    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    def get_data_loaders(self, dataset_type=DatasetType.ALL, model_type=ResNetModelType.RESNET18):

        # Create DataLoaders for training, validation, and testing
        batch_size = self.batch_size # Define your batch size
        train_loader = None
        val_loader = None
        test_loader = None

        self._apply_tranforms_from_model(model_type)

        if dataset_type == DatasetType.TRAIN_VAL:
            # Download the CIFAR-100 dataset
            train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=self._train_transform())
            # Split the training dataset into training and validation sets
            val_size = 5000  # Validation set size
            train_size = len(train_dataset) - val_size
            #use train_test_split to split the dataset from sklearn
            train_dataset, val_dataset = train_test_split(train_dataset, test_size=val_size, random_state=42)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        elif dataset_type == DatasetType.TEST:
            test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=self._test_transform())
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        else:
           # Download the CIFAR-100 dataset
            train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=self._train_transform())
            test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=self._test_transform())

            # Split the training dataset into training and validation sets
            val_size = 5000  # Validation set size
            train_size = len(train_dataset) - val_size
            #use train_test_split to split the dataset from sklearn
            train_dataset, val_dataset = train_test_split(train_dataset, test_size=val_size, random_state=42)
            
            # Create DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2) 

        # Return the DataLoaders
        return train_loader, val_loader, test_loader
    
    # Set the random seed for reproducibility
    def _train_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # Normalize using CIFAR-100 stats
        ])
    
    def _test_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # Normalize using CIFAR-100 stats
        ])

    def _apply_tranforms_from_model(self, model_type):
        if model_type == ResNetModelType.RESNET18:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            weights.transforms()
        elif model_type == ResNetModelType.RESNET34:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
            weights.transforms()
        elif model_type == ResNetModelType.RESNET50:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            weights.transforms()
        else:
            raise ValueError("Invalid ResNet model type. Choose from RESNET18, RESNET34, or RESNET50.")