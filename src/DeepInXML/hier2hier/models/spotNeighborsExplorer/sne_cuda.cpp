#include <torch/torch.h>
#include <vector>

using namespace std;
using namespace at;

// CUDA forward declarations
vector<Tensor> exploreSpotNeighborsCuda(
    tuple<Tensor, Tensor> graph,
    Tensor alreadySeenSet,
    Tensor activeNodeSet);

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// C++ interface
vector<Tensor> exploreSpotNeighbors(
    tuple<Tensor, Tensor> graph,
    Tensor alreadySeenSet,
    Tensor activeNodeSet) {
  CHECK_INPUT(get<0>(graph));
  CHECK_INPUT(get<1>(graph));
  CHECK_INPUT(alreadySeenSet);
  CHECK_INPUT(activeNodeSet);

  return exploreSpotNeighborsCuda(input, weights, bias, old_h, old_cell);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &exploreSpotNeighbors, "Explore Spot Neighbors forward (CUDA)");
}
