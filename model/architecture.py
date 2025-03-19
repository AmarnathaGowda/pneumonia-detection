import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DEVICE

class PneumoniaClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaClassifier, self).__init__()
        # Load pre-trained ResNet18 with updated weights parameter
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify the final fully connected layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
    

if __name__ == "__main__":
    
    # Create an instance of the model
    model = PneumoniaClassifier(num_classes=2).to(DEVICE)
    print(model)
    
    # Test with a dummy input (batch of 4 images, 3 channels, 224x224)
    dummy_input = torch.randn(4, 3, 224, 224).to(DEVICE)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")