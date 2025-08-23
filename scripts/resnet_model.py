"""
resnet_model.py

Defines ResNet model types and builder for CIFAR-100 experiments.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from enum import Enum
import torch.nn.init as init


class ResNetModelType(Enum):
    """
    Enum for supported ResNet model types.
    """
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"


class ResNetModelBuilder:
    """
    Builds ResNet models for CIFAR-100 classification.
    """
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type: ResNetModelType, num_classes: int, pretrained: bool = True):
        """
        Returns a ResNet model of the specified type and number of classes.
        Args:
            model_type (ResNetModelType): Type of ResNet.
            num_classes (int): Number of output classes.
            pretrained (bool): Use pretrained weights.
        Returns:
            torch.nn.Module: ResNet model.
        """
        # Initialize the ResNet model based on the specified model_type
        if model_type == ResNetModelType.RESNET18:
            if pretrained:
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet18(pretrained=False)
        elif model_type == ResNetModelType.RESNET34:
            if pretrained:
                model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet34(pretrained=False)   
        elif model_type == ResNetModelType.RESNET50:
            if pretrained:
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError("Invalid ResNet model type. Choose from RESNET18, RESNET34, or RESNET50.")
        
       
        #change to no gradient
        for param in model.parameters():
            param.requires_grad = False

        # Modify the final fully connected layer to match num_classes
        model.classifier = nn.Sequential()
        
        if (model_type == ResNetModelType.RESNET18 or model_type == ResNetModelType.RESNET34):
            model.fc = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features=512, out_features=100)
            )
        else:
            model.fc = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features=2048, out_features=1000),
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features=1000, out_features=100)
            )


        return model
