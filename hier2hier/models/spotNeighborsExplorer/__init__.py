"""
In SpotNeighborsExplorer, we explore neighbors of the attention spotlight for increasing
attention factors. If the attentoin factors are above a certain threshold, we will include
them in the next iteration of attention spotlight. 

SNE = SpotNeighborsExplorer
"""
import os
import torch.nn

from hier2hier.util import blockProfiler, methodProfiler, appendProfilingLabel
from torch.utils.cpp_extension import load

path = os.path.dirname(__file__)

from .sne_python import SpotNeighborsExplorerPy
# Only load CPP/CUDA/TS modules if needed.
SpotNeighborsExplorerCpp = None
SpotNeighborsExplorerCuda = None
SpotNeighborsExplorerTS = None

class SpotNeighborsExplorer(torch.nn.Module):
    """
    This module is part of an optimization of attention mechanism. In the regular attention
    (https://arxiv.org/abs/1502.03044), all positions in the input sequence are part of the
    linear combination. That is what is called global attention. 
    In hier2hier, sequence positions are replaced by graph positions.
    Here, we make an approximation called local attention.
    We ignore some of the sequence positions with small attention factors from consideration.

    Attention Spotlight:
        The graph positions which are currently under spotlight are collectively called
        attention spotlight.

    Spot Neighbors Exploration:
        Here we explore neighbors of the spotlight for attention factors. If the factors
        are high enough, we will include them into next spotlight. 
    """
    def __init__(self, impl_selection=None, device=None):
        global SpotNeighborsExplorerCpp, SpotNeighborsExplorerCuda, SpotNeighborsExplorerTS
        super().__init__()

        if impl_selection is None:
            impl_selection="cuda" if device is not None and device.type=="cuda" else "cpp"
        self.impl_selection = impl_selection

        if impl_selection == "python":
            self.sne_base = SpotNeighborsExplorerPy()
        elif impl_selection == "torch_script":
            if SpotNeighborsExplorerTS is None:
                from .sne_torch_script import SpotNeighborsExplorerTS
            self.sne_base = SpotNeighborsExplorerTS()
        elif impl_selection == "cpp":
            if SpotNeighborsExplorerCpp is None:
                SpotNeighborsExplorerCpp = load(
                    name="sne_cpp",
                    sources=[path + "/sne.cpp"],
                    verbose=True)
        elif impl_selection == "cuda":
            if SpotNeighborsExplorerCuda is None:
                SpotNeighborsExplorerCuda = load(
                    name='sne_cuda',
                    sources=[path+'/sne_cuda.cpp', path+'/sne_cuda_kernel.cu'],
                    extra_cuda_cflags=["-D_MWAITXINTRIN_H_INCLUDED", "-D_FORCE_INLINES", "-D__STRICT_ANSI__"],
                    verbose=True)

    @methodProfiler
    def forward(self, graph, alreadySeenSet, activeNodeSet):
        if self.impl_selection in ["python", "torch_script"]:
            return self.sne_base(graph, alreadySeenSet, activeNodeSet)
        elif self.impl_selection == "cuda":
            activeSetOut =  SpotNeighborsExplorerCuda.forward(graph, alreadySeenSet, activeNodeSet)
            activeSetOut = torch.unique(torch.sort(activeSetOut)[0])
            alreadySeenOut = torch.sort(torch.cat([activeSetOut, alreadySeenSet]))[0]
            return alreadySeenOut, activeSetOut
        elif self.impl_selection == "cpp":
            return SpotNeighborsExplorerCpp.forward(graph, alreadySeenSet, activeNodeSet)

