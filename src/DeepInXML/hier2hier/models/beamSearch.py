import torch
from hier2hier.util import (onehotencode, blockProfiler, methodProfiler, lastCallProfile,
                            batched_index_select)

def BeamSearch(
        symbolGeneratorModel,
        modelStartState,
        maxOutputLen,
        maxBeamCount,
        sos_id,
        eos_id,
        outBeamCount=1,
        device=None,
        traceStates=False,
        ):
    """
        symbolGeneratorModel:
            A model which can generate probabilities for next symbol, given current
            state and current input symbol.
        modelStartState:
            Start state for the model.
        maxOutputLen
            Maximum length of the output symbol vector.
        sos_id, eos_id
            Start and end symbols.
        outBeamCount
            Number of fully decoded outputs to generate.
    """
    if isinstance(modelStartState, tuple):
        modelStartStateTuple = modelStartState
    else:
        # If not tuple, make it a singleton tuple.
        modelStartStateTuple = (modelStartState, )
        

    curBeamCount = 1
    sampleCount = modelStartStateTuple[0].shape[0]

    # Beam probabilities.
    # Shape: SampleCount X CurBeamCount.
    # Data: Probability value.
    curBeamProbs = torch.ones(sampleCount, curBeamCount)

    # Current symbol outputs.
    # Shape: SampleCount X CurBeamCount.
    # Data: Symbol ID.
    curBeamSymbols = torch.tensor(
        [
            [ sos_id for _ in range(curBeamCount) ]
            for _ in range(sampleCount)
        ],
        device=device
    )

    # Tuple of current beam states.
    # Each tuple:
    #   Shape: SampleCount X CurBeamCount X ModelStateShape.
    #   Data: Vectors representing input state.
    curBeamStatesTuple = tuple(
        (
            None if modelStartState is None
            else (
                modelStartState.view(
                    [sampleCount, curBeamCount] 
                    + list(modelStartState.shape[1:])
                )
            )
        )
        for modelStartState in modelStartStateTuple
    )

    # List used to reconstruct the character trail in the winning beam(s).
    # Each list element
    #   Shape: SampleCount X CurBeamCount. Data: Probablity.
    #   Data: Last symbol seen at the current (sample, beam) index.
    beamSymbolsTrace = [ curBeamSymbols ]

    # List used to reconstruct the character trail in the winning beam(s).
    # Each list element is a tuple.
    #     Each tuple element:
    #         Shape: SampleCount X CurBeamCount. Data: Probablity.
    #         Data: Accumumulated states of the current beam at the
    #             current (sample, beam) index.
    beamStatesTupleTrace = [ curBeamStatesTuple ]

    # List used to reconstruct the character trail in the winning beam(s).
    # Each list element:
    #   Shape: SampleCount X CurBeamCount.
    #   Data: Index of beam in the previous list entry that this one continues.
    #         None in the first list entry as there is no back pointer.
    beamzBackPtrsTrace = [ None ]

    # Shape: SampleCount X CurBeamCount.
    # Data: True if the eos_id has been seen and it is now a closed beam.
    accumulatedClosedBeams = None

    for n in range(maxOutputLen):
        # Compute next candidate symbol probabilities.
        # nextBeamStatesTuple: Tuple of states.
        #   Each state:
        #       Shape: (SampleCount X CurBeamCount X <State Dims>
        # nextCandidateSymbolProbs
        #   Shape: SampleCount X CurBeamCount X VocabLen
        #   Data: Prob value
        (
            nextBeamStatesTuple,
            nextCandidateSymbolProbs,
        )  = symbolGeneratorModel(curBeamStatesTuple, curBeamSymbols)
        vocabLen = nextCandidateSymbolProbs.shape[-1]

        # Compute chain probabilities for next symbol.
        # Shape: SampleCount X CurBeamCount X VocabLen
        nextBeamProbs = curBeamProbs.view(sampleCount, curBeamCount, 1) * nextCandidateSymbolProbs

        # Flatten beam.
        # Shape: SampleCount X (CurBeamCount*VocabLen)
        nextBeamProbsFlat = nextBeamProbs.view(sampleCount, curBeamCount*vocabLen)

        # Pick top k best.
        # Shape: (SampleCount X self.maxBeamCount, SampleCount X self.maxBeamCount)
        (
            nextBeamProbsSelected, nextBeamzFlatBackPtrs
        ) = torch.topk(nextBeamProbsFlat, maxBeamCount)

        # Indices of the previous beam that the next beam is extending.
        # Shape: SampleCount X self.maxBeamCount
        nextBeamzBackPtrs = nextBeamzFlatBackPtrs / vocabLen

        # Last symbol seen at the current (sample, beam) index.
        # Shape: SampleCount X self.maxBeamCount
        nextBeamSymbols = nextBeamzFlatBackPtrs % vocabLen

        nextBeamProbs = batched_index_select(nextBeamProbsFlat, 1, nextBeamzFlatBackPtrs)

        # Index into nextBeamStates to get topK nextBeamStates.
        nextBeamStatesTuple = tuple (
            batched_index_select(nextBeamStates, 1, nextBeamzBackPtrs)
            for nextBeamStates in nextBeamStatesTuple
        )

        # Save beam traces to be used in decoding later.
        beamSymbolsTrace.append(nextBeamSymbols)
        beamzBackPtrsTrace.append(nextBeamzBackPtrs)
        if traceStates:
            beamStatesTupleTrace.append(nextBeamStatesTuple)

        # Check for the end.
        if accumulatedClosedBeams is None:
            accumulatedClosedBeams = (nextBeamSymbols == eos_id)
        else:
            currentlyClosedBeams = (nextBeamSymbols == eos_id)
            accumulatedClosedBeams = accumulatedClosedBeams | currentlyClosedBeams

        if all((accumulatedClosedBeams.view(-1))):
            # Break if all beams are closed
            break

        # Prepare for next iteration.
        curBeamCount = maxBeamCount
        curBeamSymbols = nextBeamSymbols
        curBeamProbs = nextBeamProbs
        curBeamStatesTuple = nextBeamStatesTuple

    # Decode accumulated beams to generate final result.
    stateCount = len(modelStartStateTuple)
    decodedSymbolBeams = []
    if traceStates:
        decodedStatesTupleBeams = [[] for _ in range(stateCount)]
    else:
        decodedStatesTupleBeams = None
    outputLen = len(beamSymbolsTrace)
    for i in range(min(outBeamCount, maxBeamCount)):
        symbolColumns = []
        if traceStates:
            stateTupleColumns = [[] for _ in range(stateCount)]
        curBeamIndices = torch.LongTensor([i for _ in range(sampleCount)])
        for j in range(outputLen-1, -1, -1):
            symbolsColumn = batched_index_select(beamSymbolsTrace[j], 1, curBeamIndices)
            symbolColumns.append(symbolsColumn)

            if traceStates:
                for k in range(stateCount):
                    stateTupleColumns[k].append (
                        batched_index_select(beamStatesTupleTrace[j][k], 1, curBeamIndices)
                    )
            if beamzBackPtrsTrace[j] is None:
                curBeamIndices = None
            else:
                curBeamIndices = batched_index_select(beamzBackPtrsTrace[j], 1, curBeamIndices)
                curBeamIndices = curBeamIndices.view(sampleCount,)

        symbolColumns.reverse()
        decodedSymbolBeams.append(torch.cat(symbolColumns, 1))

        # For each state-tuple-entry in a particular beam.
        if traceStates:
            for k in range(stateCount):
                stateTupleColumns[k].reverse()
                # Each state tuple entry has multiple columns, one for each character position.
                decodedStatesTupleBeams[k].append(
                    torch.cat(stateTupleColumns[k], 1)
                )

    # Convert all lists to tuples.
    if traceStates:
        decodedStatesTupleBeams = [
            tuple(decodedStatesTupleBeam)
            for decodedStatesTupleBeam in decodedStatesTupleBeams
        ]

    return decodedSymbolBeams, decodedStatesTupleBeams

if __name__ == '__main__':
    # Define search space and vocabulary.
    directions = torch.tensor([
                    (0.0, 0.0, 0.0), # Don't use this.
                    (-1, -1, -1), (-1, -1, +1), (-1, +1, -1), (-1, +1, +1),
                    (+1, -1, -1), (+1, -1, +1), (+1, +1, -1), (+1, +1, +1),
                    (0.0, 0.0, 0.0),
                ])
    sos_id = 0
    eos_id = len(directions)-1
    vocabLen = len(directions)
    stateVecLen = len(directions[0])

    # Define problem instance.
    if True:
        # Easy to understand, deterministic problem instance.
        step = 0.05
        maxOutputLen=20
        maxBeamCount = 5
        outBeamCount = 4

        # Target position.
        targetPosition = torch.tensor([0.5, 0.25, 0.5])

        # modelStartState = torch.rand(sampleCount, stateVecLen)
        modelStartState = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ], dtype=torch.float)
        sampleCount = len(modelStartState)
    else:
        # Randomly generated problem instance.
        step = 0.02
        maxOutputLen=15
        maxBeamCount = 5
        outBeamCount = 4

        # Random like search.
        targetPosition = torch.rand(3)

        # Give start state to each sample.
        sampleCount = 8
        modelStartState = torch.rand(sampleCount, stateVecLen) 
        
    
    initOffset = 3
    sos_burden = 10
    directions *= step
    modelStartState += initOffset
    def modOne(t):
        """
        Brings all values of a tensor into the range [0, 1).
        """
        return t
        return t - torch.floor(t)

    def testModel(beamStatesIn, lastBeamSymbolsIn):
        """
            Model to test beamSearch,
            Inputs:
                beamStatesIn:
                    Shape: sampleCount * beamCount * stateVecLen
                    Data: 3D position vector in a 1x1 cube with each of its faces joined with
                        the opposite face (Flat hyper-toroid).
                lastBeamSymbolsIn:
                    Shape: sampleCount * beamCount
                    Data: ID of the beam symbol output in the last iteration.
            Outputs:
                beamStatesOut:
                    Shape: sampleCount * beamCount * stateVecLen
                    Data: 3D position vector in a 1x1 cube with each of its faces joined with
                        the opposite face (Flat hyper-toroid).
                beamSymbolProbsOut:
                    Shape: sampleCount * beamCount * vocabLen
                    Data: Values to be treated as log probabilities.
                          Computed(for testing) as modulo distances of new state from target-point.
        """
        if isinstance(beamStatesIn, tuple):
            beamStatesIn = beamStatesIn[0]
        sampleCount, beamCount = lastBeamSymbolsIn.shape
        stateNeedsInit = (lastBeamSymbolsIn == sos_id).type(torch.float).view(sampleCount, beamCount, 1)
        initializedStates = beamStatesIn - initOffset
        adjustedStates = modOne(beamStatesIn + directions[lastBeamSymbolsIn])
        beamStatesOut = (1-stateNeedsInit) * adjustedStates + stateNeedsInit * initializedStates
        # assert(all(abs(beamStatesOut.view(-1)) <= 1))

        # Create repeataed version of beamStatesOut, one repetition for each vocab symbol.
        beamStatesOutRepeated = beamStatesOut.view(sampleCount, beamCount, 1, stateVecLen)
        beamStatesOutRepeated = beamStatesOutRepeated.repeat(1, 1, vocabLen, 1)

        # Compute distances from targetPosition when different directions are chosen.
        beamSymbolProbsOut = (
            torch.norm(modOne(beamStatesOutRepeated + directions - targetPosition), 2, -1)
            - torch.norm(modOne(beamStatesOutRepeated-targetPosition), 2, -1)
        )

        # Weigh down sos_id. It is always low priority.
        beamSymbolProbsOut[..., sos_id] += sos_burden

        # Convert to probabilities. More distance => Low probability.
        beamSymbolProbsOut = torch.exp(-beamSymbolProbsOut)
        beamSymbolProbsOut /= torch.sum(beamSymbolProbsOut, -1).view(sampleCount, beamCount, 1)

        beamStatesTupleOut = (beamStatesOut, )
        return beamStatesTupleOut, beamSymbolProbsOut

    decodedSymbolBeams, decodedStatesTupleBeams = BeamSearch(
            testModel,
            modelStartState,
            maxOutputLen,
            maxBeamCount,
            sos_id,
            eos_id,
            outBeamCount=outBeamCount,
            traceStates=True,
        )

    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Data for a three-dimensional line
    #zline = np.linspace(0, 15, 1000)
    #xline = np.sin(zline)
    #yline = np.cos(zline)
    #ax.plot3D(xline, yline, zline, 'gray')

    # Data for three-dimensional scattered points
    #zdata = 15 * np.random.random(100)
    #xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
    #ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
    #ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');

    bestBeamStatesTuple = decodedStatesTupleBeams[0]
    bestBeamStates = bestBeamStatesTuple[0]
    colors = ("red", "green", "blue", "orange", "purple", "gray", "violet", "yellow", "lightblue")
    for i in range(sampleCount):
        xdata = bestBeamStates[i][1:, 0].numpy()
        ydata = bestBeamStates[i][1:, 1].numpy()
        zdata = bestBeamStates[i][1:, 2].numpy()
        ax.plot3D(xdata, ydata, zdata, c=colors[i]);
    plt.show()
