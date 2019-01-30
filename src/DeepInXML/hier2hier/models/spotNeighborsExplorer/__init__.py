"""
SNE = SpotNeighborsExplorer
"""
import os
import torch.nn

from hier2hier.util import blockProfiler, methodProfiler, appendProfilingLabel
from torch.utils.cpp_extension import load

path = os.path.dirname(__file__)

from .sne_python import SpotNeighborsExplorerPy
SpotNeighborsExplorerCpp = None
SpotNeighborsExplorerCuda = None
SpotNeighborsExplorerTS = None

class SpotNeighborsExplorer(torch.nn.Module):
    def __init__(self, impl_selection=None, device=None):
        global SpotNeighborsExplorerCpp, SpotNeighborsExplorerCuda, SpotNeighborsExplorerTS
        super().__init__()

        if impl_selection is None:
            impl_selection="cuda" if device is not None and device.type=="cuda" else "cpp"
        self.impl_selection = impl_selection

        if impl_selection == "python":
            self.sne_base = SpotNeighborsExplorerPy()
        elif impl_selection == "cpp":
            if SpotNeighborsExplorerCpp is None:
                SpotNeighborsExplorerCpp = load(name="sne_cpp", sources=[path + "/sne.cpp"], verbose=True)
            self.sne_base = SpotNeighborsExplorerCpp
        elif impl_selection == "cuda":
            if SpotNeighborsExplorerCuda is None:
                SpotNeighborsExplorerCuda = load(name='sne_cuda', sources=[path+'/sne_cuda.cpp', path+'/sne_cuda_kernel.cu'], verbose=True)
            self.sne_base = SpotNeighborsExplorerCuda
        elif impl_selection == "torch_script":
            if SpotNeighborsExplorerTS is None:
                from .sne_torch_script import SpotNeighborsExplorerTS
            self.sne_base = SpotNeighborsExplorerTS()

    @methodProfiler
    def forward(self, *argc, **kargv):
        if self.impl_selection in ["python", "torch_script"]:
            return self.sne_base(*argc, **kargv)
        else:
            return self.sne_base.forward(*argc, **kargv)

