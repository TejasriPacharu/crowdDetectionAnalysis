import torch
import torch.nn as nn
import torchvision.models as models


class CSRNet(nn.Module):
    """
    CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes
    
    This model uses VGG16 as the frontend and dilated convolutions as the backend
    for crowd density estimation.
    """
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        
        # Load pretrained VGG16 as the frontend
        vgg16_bn = models.vgg16_bn(pretrained=load_weights)
        self.frontend_features = nn.Sequential(*list(vgg16_bn.features.children())[:33])
        
        # Backend dilated convolutions
        self.backend_features = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        
        # Output layer to generate density map
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
        )
        
        # Initialize weights for the backend and output layer
        self._initialize_weights(self.backend_features)
        self._initialize_weights(self.output_layer)
    
    def forward(self, x):
        # Frontend feature extraction (VGG16)
        x = self.frontend_features(x)
        
        # Backend feature extraction (dilated convolutions)
        x = self.backend_features(x)
        
        # Output density map
        x = self.output_layer(x)
        
        return x
    
    def _initialize_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def csrnet(pretrained=False):
    """
    Create a CSRNet model
    
    Args:
        pretrained: Whether to use pretrained VGG16 weights
        
    Returns:
        CSRNet model
    """
    model = CSRNet(load_weights=pretrained)
    return model
