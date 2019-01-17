"""
SNE = SpotNeighborsExplorer
"""
import os
from torch.utils.cpp_extension import load
import torch.nn

from hier2hier.util import blockProfiler, methodProfiler, appendProfilingLabel

path = os.path.dirname(__file__)

impl_selection = "cpp"

if impl_selection == "cpp":
    sne_cpp = load(name="sne_cpp", sources=[path + "/sne.cpp"], verbose=True)
    class SpotNeighborsExplorer(torch.nn.Module):
        def __init__(self):
            super().__init__()

        @methodProfiler
        def wrapper(self, *argc, **kargv):
            return self(*argc, **kargv)

        def forward(self, *argc, **kargv):
            return sne_cpp.forward(*argc, **kargv)

elif impl_selection == "cuda":
    sne_cuda = load(name='sne_cuda', sources=[path+'/sne_cuda.cpp', path+'/sne_cuda_kernel.cu'], verbose=True)

else:
    from hier2hier.models.spotNeighborsExplorer.sne_torch_script import SpotNeighborsExplorer

