import torch
from torch import nn
from torch.nn import functional as F







class Basic_CNN_Model(nn.Module):
    def __init__(self, input_size: 'tuple[int]', num_classes: int, cnn_out_channels1: int, cnn_out_channels2: int, kernel_size,
                fc_hidden_layer_size: int):
        super(Basic_CNN_Model, self).__init__()
        self.cnn_l1 = nn.Conv2d(
            in_channels = 3, 
            out_channels = cnn_out_channels1, 
            kernel_size = kernel_size, 
            padding='same')
        self.maxpool1 = nn.MaxPool2d(kernel_size, padding='same')
        self.cnn_l2 = nn.Conv2d(
            in_channels = cnn_out_channels1, 
            out_channels = cnn_out_channels2, 
            kernel_size = kernel_size, 
            padding='same')
        self.maxpool2 = nn.MaxPool2d(kernel_size, padding='same')
        f_in = cnn_out_channels2 * input_size[0] * input_size[1]
        self.fc1 = nn.Linear(f_in, fc_hidden_layer_size)
        self.fc2 = nn.Linear(fc_hidden_layer_size, num_classes)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.cnn_l1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.cnn_l2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


