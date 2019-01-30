#include <vector>
#include <set>
#include <algorithm>
#include <torch/extension.h>
#include <iostream>

using namespace std;
using namespace at;

template<class AccessorT, class ObjT>
int accessor_binary_search(const AccessorT &accessor, const ObjT &search)  
{
    auto first = 0, last = int(accessor.size(0)) - 1;
    int middle = (first+last)/2;
    
    while (first <= last)
    {
        if (accessor[middle] < search)
            first = middle + 1;    
        else if (accessor[middle] == search)
           return middle;
        else
           last = middle - 1;
                                    
        middle = (first + last)/2;
    }
    return -1;  
}

vector<Tensor> exploreSpotNeighbors(
    tuple<Tensor, Tensor> graph, 
    Tensor alreadySeenSet,
    Tensor activeNodeSet
    ) {

    auto graphNbrs = get<0>(graph);
    auto graphNbrsCount = get<1>(graph);

    // Create accessors for input tensors' data. 
    auto accessAlreadySeenSet = alreadySeenSet.accessor<int64_t,1>();
    auto accessActiveNodeSet = activeNodeSet.accessor<int64_t,1>();
    auto accessGraphNbrs = graphNbrs.accessor<int64_t,2>();
    auto accessGraphNbrsCount = graphNbrsCount.accessor<int64_t,1>();

    set<int64_t> neighborSet;
    // cout <<"\nStart CPPCode\n";
    for(auto nodeIndex=0;nodeIndex<accessActiveNodeSet.size(0);nodeIndex++) {
        auto node = accessActiveNodeSet[nodeIndex];
        auto nbrCount = accessGraphNbrsCount[node];
        auto curNbrs=accessGraphNbrs[node];
        // cout <<"\tFor "<<node<<"\n";
        for(int nbrIndex=0;nbrIndex<nbrCount;nbrIndex++)
        {
            auto nbr=curNbrs[nbrIndex];
            bool found = accessor_binary_search(accessAlreadySeenSet, nbr) >= 0;
            // cout <<(found?" Found ":" Not Found ") << nbr;
            if (!found)
            {
                neighborSet.insert(nbr);
            }
        }
    }
    auto activeNodeSetOut = empty({int64_t(neighborSet.size())}, torch::CPU(kLong));
    accessActiveNodeSet = activeNodeSetOut.accessor<int64_t,1>();
    int i=0;
    for(auto iter=neighborSet.begin();iter!=neighborSet.end();iter++, i++){
        accessActiveNodeSet[i] = *iter; 
    }

    auto alreadySeenSetOut = get<0>(sort(cat({ alreadySeenSet, activeNodeSetOut })));

    // cout <<"End CPPCode\n";
    // cout.flush();
    return {alreadySeenSetOut, activeNodeSetOut};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &exploreSpotNeighbors, "Explore Spot Neighbors forward");
}
