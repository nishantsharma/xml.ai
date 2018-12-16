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
