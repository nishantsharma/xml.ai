/*
    CPP component of CUDA implementaiton of Spot Neighbors Explorer.

    In SpotNeighborsExplorer, we explore neighbors of the attention spotlight for increasing
    attention factors. If the attentoin factors are above a certain threshold, we will include
    them in the next iteration of attention spotlight. 
*/
#include <torch/extension.h>
#include <vector>
#include <thrust/tuple.h>

using namespace std;
using namespace at;

// CUDA forward declarations
thrust::tuple<Tensor, int> exploreSpotNeighborsCuda(
    tuple<Tensor, Tensor> graph,
    Tensor alreadySeenSet,
    Tensor activeNodeSet);

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// C++ interface
Tensor exploreSpotNeighbors(
  		  tuple<Tensor, Tensor> graph,
  		  Tensor alreadySeenSet,
  		  Tensor activeNodeSet) {
    CHECK_INPUT(get<0>(graph));
    CHECK_INPUT(get<1>(graph));
    CHECK_INPUT(alreadySeenSet);
    CHECK_INPUT(activeNodeSet);

    // Find neighbors using CUDA.
    auto neighborsInfoFound = exploreSpotNeighborsCuda(graph, alreadySeenSet, activeNodeSet);
    // cout <<"\nAfter CUDA call.";cout.flush();
    auto neighborsFound = get<0>(neighborsInfoFound);
    auto neighborsFoundCount = get<1>(neighborsInfoFound);

    // Remove duplicates.
    neighborsFound = neighborsFound.narrow(0, 0, neighborsFoundCount);
    // cout <<"\nAfter narrow."<<neighborsFoundCount;cout.flush();
    return neighborsFound;

    // Prepare and return result.
    //auto activeNodeSetOut = get<0>(sort(neighborsFound));
    //auto alreadySeenSetOut = get<0>(sort(cat({ alreadySeenSet, neighborsFound })));
    // cout <<"\nAfter merge.";cout.flush();

    //return { alreadySeenSetOut, activeNodeSetOut };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &exploreSpotNeighbors, "Explore Spot Neighbors forward (CUDA)");
}
