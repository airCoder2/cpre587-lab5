
#include <iostream>
#include <cmath>
#include <algorithm>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"
#include "Flatten.h"

namespace ML
{

    // --- Begin Student Code ---

    // ME: the LayerData object passed has been loaded before (by img.loadData() no need to do it again)
    void FlattenLayer::computeNaive(const LayerData &dataIn, const int layer_num) const
    {
        // flatten was one line, because I added a new = operator that takes care of it in LayerData object
        getOutputData() = dataIn;
    }

    void FlattenLayer::computeAccelerated(const LayerData &dataIn, const int layer_num) const
    {
        computeNaive(dataIn, layer_num);
    }

    void FlattenLayer::computeThreaded(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
    }

    void FlattenLayer::computeTiled(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
    }

    void FlattenLayer::computeSIMD(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
    }
} // namespace ML
