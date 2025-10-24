
import torch
import torch.nn as nn

class FullyConnectedNet(nn.Module):
    """A 3-hidden-layer fully connected neural network with BatchNorm and Dropout."""
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate):
        super(FullyConnectedNet, self).__init__()
        
        layers = []
        in_size = input_size
        
        for h_size in hidden_layers:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.BatchNorm1d(h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_size = h_size
            
        layers.append(nn.Linear(in_size, output_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the image
        x = x.view(x.size(0), -1)
        return self.network(x)
