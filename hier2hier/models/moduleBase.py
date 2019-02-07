"""
Implements base class for all modules implemented within the Hier2hier scope.
"""
import torch
import torch.nn as nn
        
class ModuleBase(nn.Module):
    def __init__(self, schemaVersion, device):
        super().__init__()
        assert(schemaVersion is not None)
        self.set_device(device)
        self.schemaVersion = schemaVersion

    def set_device(self, device):
        self.device = device
        for child in self.children():
            if isinstance(child, ModuleBase):
                child.set_device(device)

    def reset_parameters(self, device):
        raise NotImplementedError("Param initialization")
        for param in model.parameters():
            param.data.uniform_(-1.0, 1.0)

    def upgradeSchema(self, newSchemaVersion):
        # Migrate schema of all children.
        for childModule in self.children():
            if isinstance(childModule, ModuleBase) or isinstance(childModule, torch.jit.ScriptModule):
                newModelArgs = childModule.upgradeSchema(newSchemaVersion)

        # Migrating self.
        curSchemaVersion = -1 if not hasattr(self, "schemaVersion") else self.schemaVersion
        assert(curSchemaVersion <= newSchemaVersion)
        while curSchemaVersion != newSchemaVersion:
            nextSchemaVersion = curSchemaVersion+1
            self.singleStepSchema(nextSchemaVersion)
            self.schemaVersion = nextSchemaVersion
            curSchemaVersion = nextSchemaVersion

    def singleStepSchema(self, schemaVersion):
        if schemaVersion is 0:
            return
        else:
            raise NotImplementedError("Schema migration should be overridden in the derived module.")

    def reconfigure(self, newModelArgs, debug):
        if schemaVersion is 0:
            return newModelArgs
        else:
            raise NotImplementedError("Schema migration should be overridden in the derived module.")
