import torch
import torch.nn as nn
import torchvision.models as models


class DensityClassifier(nn.Module):
    """
    Density Classifier to determine if a crowd is dense or sparse
    
    This model uses ResNet18 as the backbone and outputs a binary classification:
    0: Sparse crowd (use YOLOv8)
    1: Dense crowd (use CSRNet)
    """
    def __init__(self, pretrained=True):
        super(DensityClassifier, self).__init__()
        
        # Load pretrained ResNet18 as the backbone
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add a new fully connected layer for binary classification
        self.fc = nn.Linear(512, 1)
        
        # Initialize the weights of the new layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        x = self.backbone(x)
        
        # Flatten the features
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.fc(x)
        
        # Apply sigmoid to get probability
        x = torch.sigmoid(x)
        
        return x


def density_classifier(pretrained=True):
    """
    Create a density classifier model
    
    Args:
        pretrained: Whether to use pretrained ResNet18 weights
        
    Returns:
        Density classifier model
    """
    model = DensityClassifier(pretrained=pretrained)
    return model
