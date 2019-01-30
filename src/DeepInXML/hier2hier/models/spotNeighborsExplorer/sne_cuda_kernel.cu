#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <thrust/tuple.h>
#include <thrust/unique.h>
#include <thrust/sort.h>

using namespace std;
using namespace at;

template <class T>
__device__ __forceinline__ int device_binary_search(const T* __restrict__ startPtr, const T* __restrict__ endPtr, T value)
{
    auto last = (endPtr - startPtr) - 1;
    decltype(last) first = 0;
    auto middle = (first+last)/2;

    while (first <= last)
    {
        if (startPtr[middle] < value)
            first = middle + 1;
        else if (startPtr[middle] == value)
            return middle;
        else
            last = middle - 1;

        middle = (first + last)/2;
    }
    return -1;
}

__device__ __forceinline__ int prefixsum(int threadId, int itemCountForThread)
{
    int threadsPerBlock = blockDim.x * blockDim.y;
    __shared__ int temp[4000];

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

template <typename scalar_t>
__device__ __forceinline__ int getNbr(
		int activeNodeIndex,
		int nbrIndex,
		const scalar_t* __restrict__ activeNodeSet,
     	        const scalar_t* __restrict__ graphNbrs,
		const scalar_t* __restrict__ graphNbrsCount,
                size_t graphMaxNbrCount) {
    // Get node index.
    scalar_t nodeIndex = activeNodeSet[activeNodeIndex];
    
    // Obtain and validate nbrIndex.
    auto nbrCount = graphNbrsCount[nodeIndex];
    if(nbrIndex >= nbrCount) {
        // printf("\nTID:%d, bIdx(%d,%d), tIdx(%d,%d), nodeIndex=%d, nbrIndex=%d >= nbrCount=%d",
        //		threadId, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, nodeIndex,
        //		nbrIndex, nbrCount);  
        // insertCountForThread = 0;
        return -1;
    } else {
        // Check if the nbr needs to be inserted into the activeNodeSet.
        auto nbrOffset = nodeIndex*graphMaxNbrCount + nbrIndex;
        return graphNbrs[ nbrOffset ];
    }
}

template <typename scalar_t>
__device__ __forceinline__ void writeNbr(scalar_t* __restrict__ nbrsFound, int offset, scalar_t nbr) {
    nbrsFound[offset] = nbr;
}

template<typename scalar_t, typename size_type>
__global__ void exploreSpotNeighborsKernel(
            const scalar_t* __restrict__ activeNodeSet,
            size_type activeNodeCount,
            const scalar_t* __restrict__ graphNbrs,
            const scalar_t* __restrict__ graphNbrsCount,
            const scalar_t* __restrict__ alreadySeenSet,
            size_type alreadySeenCount,
            scalar_t* __restrict__ nbrsFound,
            size_t graphNodeCount,
            size_t graphMaxNbrCount,
            int* __restrict__ nbrsFoundCountDevice
) {
    // All threads in the block add their offsets to offsetForBlock.
    __shared__ int offsetForBlock;

    // Thread ID within block.
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    // int blockId = blockIdx.y * gridDim.x + blockIdx.x;

#if 0
    if (threadId == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    {
        for(int i=0;i<graphNodeCount;i++)
            for(int j=0;j<graphMaxNbrCount;j++) {
                 auto nbrOffset = i*graphMaxNbrCount + j;
                 auto nbr = graphNbrs[ nbrOffset ];
                 printf("\n\tGraph value %d, %d, nbr@%d=%d", i, j, nbrOffset, nbr);
            }
    }
    __syncthreads();
    return;
#endif

    // Obtain and validate nodeIndex.
    int activeNodeIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int nbrIndex = blockIdx.y * blockDim.y + threadIdx.y;

    int insertCountForThread = 1;
    int nbr = -1; 
    if(activeNodeIndex >= activeNodeCount || nbrIndex >= graphMaxNbrCount) {  
        insertCountForThread = 0;
	//printf("\nBID:%d, TID:%d QUiting early, activeIndex=%d nbrIndex=%d", blockId, threadId,
	//		activeNodeIndex, nbrIndex);
    } else {
	int foundOffset = -2;
	nbr = getNbr(activeNodeIndex, nbrIndex, activeNodeSet, graphNbrs, graphNbrsCount, graphMaxNbrCount);
	if (nbr == -1) {
	    insertCountForThread = 0;
	} else {
	    foundOffset=device_binary_search(alreadySeenSet, alreadySeenSet+alreadySeenCount, scalar_t(nbr));
            if(foundOffset >= 0) {
		//printf("\n\tTID:%d Still zeroing, activeIndex=%d nbrIndex=%d, nbr=%d", threadId,
		//	       	activeNodeIndex, nbrIndex, nbr);
                // Already present. No need to insert.
                insertCountForThread = 0;
	    }
	}
	 
	if (insertCountForThread)
       	{
            //printf("\nTID:%d, bIdx(%d,%d), tIdx(%d,%d), activeIndex=%d nbrIndex=%d, nbr=%d, found@%d, So:%d",
            //    threadId, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, activeNodeIndex,
	    //    nbrIndex, nbr, foundOffset, insertCountForThread);
	}
    }

    // Compute an offset for current thread, unique among every thread in the block. 
    int offsetForThreadInBlock = prefixsum(threadId, insertCountForThread);

    //__syncthreads();
    // if (insertCountForThread)printf("\n\tB:BID:%d TID:%d, nbr=%d, offInBlock=%d", blockId, threadId, nbr, offsetForThreadInBlock);

    // Next, we compute the offset for the entire block. This is done, in only one thread, but is available
    // to every thread in block due to the shared nature of offsetForBlock.
    if(threadId == blockDim.x * blockDim.y -1)
    {
       int insertCountForBlock = offsetForThreadInBlock + insertCountForThread;
       offsetForBlock = atomicAdd(nbrsFoundCountDevice, insertCountForBlock); // get a location to write them out
    }

#if 0
    __syncthreads();
    if (nbr>=0)
    {
        printf("\nBID:%d, TID:%d, nbrsFound:%d, nbr=%d, doInsert=%d, blockOffset=%d, threadOffset=%d, insertCountForThread=%d",
             blockId, threadId, nbrsFoundCountDevice, nbr, insertCountForThread,
             offsetForBlock, offsetForThreadInBlock, insertCountForThread);
    }
#endif

    // ensure offsetForBlock is available to all threads.
    __syncthreads();

    // Finally, insert into the global activeNodeSet.
    if(insertCountForThread) 
    {
       writeNbr(nbrsFound, offsetForBlock+offsetForThreadInBlock, scalar_t(nbr));
    }
}

template<typename scalar_t, typename size_type>
__global__ void compactSpotNeighborsKernel(
            scalar_t* __restrict__ nbrsFound,
            int* __restrict__ nbrsFoundCountDevice
) {
    //thrust::device_ptr<scalar_t> allNbrsPtr(nbrsFound);
    //thrust::sort(allNbrsPtr, allNbrsPtr + *nbrsFoundCountDevice);
    //*nbrsFoundCountDevice = thrust::unique(allNbrsPtr, allNbrsPtr + *nbrsFoundCountDevice) - allNbrsPtr;
}


thrust::tuple<Tensor, int> exploreSpotNeighborsCuda(
    std::tuple<Tensor, Tensor> graph,
    Tensor alreadySeenSet,
    Tensor activeNodeSet
    ) {
    typedef decltype(alreadySeenSet.size(0)) size_type;
    // Expand input graph tuple.
    auto graphNbrs = get<0>(graph);
    auto graphNbrsCount = get<1>(graph);

    // For discovering new neighbors, create an empty tensor with size matching neighborSet.
    auto nbrsFound = at::empty({int(graphNbrs.size(0) * graphNbrs.size(1))}, graphNbrsCount.type());
    int nbrsFoundCountHost=0;
    int* nbrsFoundCountDevice;

    //cout <<"\nBefore copy";cout.flush();
    cudaMalloc(&nbrsFoundCountDevice, sizeof(nbrsFoundCountHost));	
    cudaMemcpy(nbrsFoundCountDevice, &nbrsFoundCountHost, sizeof(nbrsFoundCountHost), cudaMemcpyHostToDevice);
    //cout <<"\nAfter copy";cout.flush();

    // Launch GPU code for discovering new neighbors.
    // Number of threads.
    // dim3 threadsPerBlock(256, 4);
    dim3 threadsPerBlock(16, 2);

    // Thread block dimensions.
    const int m = (graphNbrs.size(0)+threadsPerBlock.x-1)/threadsPerBlock.x; 
    const int n = (graphNbrs.size(1)+threadsPerBlock.y-1)/threadsPerBlock.y; 
    // If graph is M x N, the blocks are M/TPB.x X N/TPB.y.
    const dim3 numBlocks(m, n);

    // Allocate shared memory.
    auto sharedMemory = 2 * blockDim.x * blockDim.y * sizeof(int);

    // Dispatch GPU kernel to spot neighborhood kernels.
    //cout <<"\nBefore kernel numBlocks=("<<m<<", "<<n<<"),"
    //     <<" threadsPerBlock=("<<threadsPerBlock.x<<", "<<threadsPerBlock.y<<")";cout.flush();
    AT_DISPATCH_INTEGRAL_TYPES(
        activeNodeSet.type(),
        "exploreSpotNeighborsKernel",
        ([&] {
            exploreSpotNeighborsKernel<scalar_t, size_type>
                           <<<numBlocks, threadsPerBlock, sharedMemory>>>(
                activeNodeSet.data<scalar_t>(),
                activeNodeSet.size(0),
                graphNbrs.data<scalar_t>(),
                graphNbrsCount.data<scalar_t>(),
                alreadySeenSet.data<scalar_t>(),
                alreadySeenSet.size(0),
                nbrsFound.data<scalar_t>(),
                graphNbrs.size(0),
                graphNbrs.size(1),
                nbrsFoundCountDevice
            );
            cudaMemcpy(&nbrsFoundCountHost, nbrsFoundCountDevice, sizeof(nbrsFoundCountHost), cudaMemcpyDeviceToHost);
        })
    );
	    //if false compactSpotNeighborsKernel<scalar_t, size_t><<<1024>>>(nbrsFound.data<scalar_t>(), nbrsFoundCountDevice);

    //cout <<"\nAfter kernel: "<<nbrsFoundCountHost;cout.flush();
    return thrust::make_tuple(nbrsFound, nbrsFoundCountHost);
}


