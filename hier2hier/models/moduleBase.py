"""
Implements base class for all modules implemented within the Hier2hier scope.
"""
import torch.nn as nn
        
class ModuleBase(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.set_device(device)

    def set_device(self, device):
        self.device = device
        for child in self.children():
            if isinstance(child, ModuleBase):
                child.set_device(device)

    def reset_parameters(self, device):
        raise NotImplementedError("Param initialization")
        for param in model.parameters():
            param.data.uniform_(-1.0, 1.0)
