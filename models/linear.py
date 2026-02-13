import torch.nn as nn

class SoftmaxLinear(nn.Module):
    def __init__(self, input_dim=3072, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)
