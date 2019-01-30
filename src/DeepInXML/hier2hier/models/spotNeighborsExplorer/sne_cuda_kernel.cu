#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <thrust>

using namespace std;
using namespace at;
using namespace thrust;

__device__ int prefixsum(int threadId, int itemCountForThread)
{
    int threadsPerBlock = blockDim.x * blockDim.y;
    __shared__ int temp[threadsPerBlock*2];

    int pout = 0;
    int pin = 1;

    if(threadId == threadsPerBlock-1)
        temp[0] = 0;
    else
        temp[threadId+1] = itemCountForThread;

    __syncthreads();

    for(int offset = 1; offset<threadsPerBlock; offset<<=1) {
       pout = 1 - pout;
       pin = 1 - pin;

       if(threadId >= offset)
           temp[pout * threadsPerBlock + threadId] = temp[pin * threadsPerBlock + threadId]
               + temp[pin * threadsPerBlock + threadId - offset];
       else
           temp[pout * threadsPerBlock + threadId] = temp[pin * threadsPerBlock + threadId];

       __syncthreads();
    }

    return temp[pout * threadsPerBlock + threadId];
}

template<class scalar_t>
__global__ void exploreSpotNeighborsKernel(
            const scalar_t* __restrict__ activeNodeSet,
            int activeNodeCount,
            const scalar_t* __restrict__ graphNbrs,
            const scalar_t* __restrict__ graphNbrsCount,
            const scalar_t* __restrict__ alreadySeenSet,
            int alreadySeenCount,
            scalar_t* neighborsFound,
            int &neighborsFoundCount
) {
    // All threads in the block add their offsets to offsetForBlock.
    __shared__ int offsetForBlock;

    // Obtain and validate nodeIndex.
    int nodeIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(nodeIndex+i >= activeNodeCount) {
        continue;
    }

    // Obtain and validate nbrIndex.
    int nbrIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if(nbrIndex >= nbrCount) {
        continue;
    }

    // Check if the nbr needs to be inserted into the activeNodeSet.
    auto nbrCount = graphNbrsCount[nodeIndex];
    auto nbr = graphNbrs[nodeIndex][nbrIndex];
    scalar_t insertCountForThread = 0;
    if (!binary_search(alreadySeenSet, alreadySeenSet+alreadySeenCount, nbr))
    {
        // Not already present.
        insertCountForThread = 1;
    }

    // Compute an offset for current thread, unique among every thread in the block. 
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int offsetForThreadInBlock = prefixsum(threadId, insertCountForThread);

    // Next, we compute the offset for the entire block. This is done, in only one thread, but is available
    // to every thread in block due to the shared nature of offsetForBlock.
    if(threadId == blockDim.x * blockDim.y - 1)
    {
       int insertCountForBlock = offsetForThreadInBlock + insertCountForThread;
       offsetForBlock = atomicAdd(neighborsFoundCount, insertCountForBlock); // get a location to write them out
    }

    // ensure offsetForBlock is available to all threads.
    __syncthreads();

    // Finally, insert into the global activeNodeSet.
    if(insertCountForThread) 
    {
       activeNodeSet[offsetForBlock + offsetForThreadInBlock] = nbr;
    }
}

vector<Tensor> exploreSpotNeighbors(
    tuple<Tensor, Tensor> graph,
    Tensor alreadySeenSet,
    Tensor activeNodeSet
    ) {
    // Expand input graph tuple.
    auto graphNbrs = get<0>(graph);
    auto graphNbrsCount = get<1>(graph);

    // For discovering new neighbors, create an empty tensor with size matching neighborSet.
    auto neighborsFound = zeros_like(graphNbrsCount);
    int &neighborsFoundCount = 0;

    // Launch GPU code for discovering new neighbors.
    // Number of threads.
    dim3 threadsPerBlock = (256, 4);

    // Number of neighbor indices handled by each thread. 
    const int k = 4;

    // Thread block dimensions.
    const int m = (graphNbrs.size(0)+threadsPerBlock.x-1)/threadsPerBlock.x; 
    const int n = (graphNbrs.size(1)+threadsPerBlock.y-1)/threadsPerBlock.y; 
    // If graph is M x N, the blocks are M/TPB.x X N/TPB.y.
    const dim3 numBlocks(m, n);

    // Dispatch GPU kernel to spot neighborhood kernels.
    AT_DISPATCH_INTEGRAL_TYPES(neighborSet.type(), "exploreSpotNeighborsKernel", ([&] {
        exploreSpotNeighborsKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
            activeNodeSet.data<scalar_t>(),
            activeNodeSet.size(0),
            graphNbrs.data<scalar_t>(),
            graphNbrsCount.data<scalar_t>(),
            alreadySeenSet.data<scalar_t>(),
            alreadySeenSet.size(0),
            neighborsFound.data<scalar_t>(),
            &neighborsFoundCount
        )
    }))

    // Remove duplicates.
    auto nbrsStart = neighborsFound.data<scalar_t>();
    auto nbrsEnd = neighborsFound.data<scalar_t>()+neighborsFoundCount;
    sort(nbrsStart, nbrsEnd);
    nbrsEnd = unique(nbrsStart, nbrsEnd);
    neighborsFoundCount = nbrsStart-nbrsEnd;
    neighborsFound = neighborsFound.narrow(0, 0, neighborsFound);

    // Prepare and return result.
    auto activeNodeSetOut = neighborsFound;
    auto alreadySeenSetOut = get<0>(sort(cat({ alreadySeenSet, neighborsFound })));
    return {alreadySeenSetOut, activeNodeSetOut};
}

#if 0
template<class scalar_t>
__global__ void exploreSpotNeighborsKernel(
            const scalar_t* __restrict__ activeNodeSet,
            int activeNodeCount,
            const scalar_t* __restrict__ graphNbrs,
            const scalar_t* __restrict__ graphNbrsCount,
            const scalar_t* __restrict__ alreadySeenSet,
            int alreadySeenCount,
            scalar_t* neighborsFound,
            int &neighborsFoundCount
) {
    // If graph is M x N, the blocks are M*k/T X (N/k).
    __shared__ int offsetForBlock;

    int nodeIndexStart = blockIdx.x * blockDim.x + threadIdx.x;
    int nodeIndexCount = 1;

    int nbrIndexStart = blockIdx.y * blockDim.y + threadIdx.y;
    int nbrIndexCount = 1;

    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Thread local memory for insertion.
    scalar_t toInsert[nodeIndexCount*nbrIndexCount];
    auto insertCountForThread = 0;

    for(auto i=0;i<nodeIndexCount;i+=1) {
        if(nodeIndexStart+i >= activeNodeCount) {
            continue;
        }
        auto nodeIndex = nodeIndexStart+i;
        auto nbrCount = graphNbrsCount[nodeIndex];
        auto curNbrs=graphNbrs[nodeIndex];
        for(auto j=0;j<nbrIndexCount;j+=1) {
            auto nbrIndex = nbrIndexStart+i;
            if(nbrIndex >= nbrCount) {
                continue;
            }
            auto nbr = curNbrs[nbrIndex];
            if (binary_search(alreadySeenSet, alreadySeenSet+alreadySeenCount, nbr))
            {
                // Not already present. Do insert.
                toInsert[insertCountForThread] = nbr; 
                insertCountForThread+=1;
            }
        }
    }

    // Find out exact number of elements discovered by this thread. Remove duplicates.
    insertCountForThread = unique(toInsert, toInsert + insertCountForThread) - toInsert;

    // Compute an offset for current thread, unique among every thread in the block. 
    int offsetForThreadInBlock = prefixsum(threadId, insertCountForThread, );

    // Next, we compute the offset for the entire block. This is done, in only one thread, but is available
    // to everyone due to the shared nature of offsetForBlock.
    if(threadId == threadsPerBlock.x * threadsPerBlock.y - 1)
    {
       int insertCountForBlock = offsetForThreadInBlock + insertCountForThread;
       offsetForBlock = atomicAdd(neighborsFoundCount, insertCountForBlock); // get a location to write them out
    }

    __syncthreads(); // ensure offsetForBlock is available to all threads.

    for(int i=0;i<insertCountForThread;i++) 
    {
       activeNodeSet[offsetForBlock + offsetForThreadInBlock + i] = toInsert[i];
    }
}
#endif
