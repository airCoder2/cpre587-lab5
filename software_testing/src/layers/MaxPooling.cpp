#include "MaxPooling.h"

#include <iostream>
#include <cmath>
#include <algorithm>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML
{

    inline size_t get_input_flat_idx(size_t height_idx, size_t width_idx, size_t depth_idx, size_t image_w, size_t image_d){
        return image_w * image_d * height_idx + image_d * width_idx + depth_idx;
    }

    inline size_t get_out_flat_idx(size_t height_idx, size_t width_idx, size_t depth_idx, size_t out_w, size_t out_d){
        return out_w * out_d * height_idx + out_d * width_idx + depth_idx;
    }

    // --- Begin Student Code ---

    // Compute the convultion for the layer data
    // ME: the LayerData object passed has been loaded before (by img.loadData() no need to do it again)
    void MaxPoolingLayer::computeNaive(const LayerData &dataIn, const int layer_num) const
    {
        size_t input_h = getInputParams().dims[0];
        size_t input_w = getInputParams().dims[1];
        size_t input_d = getInputParams().dims[2];

        size_t output_w = getOutputParams().dims[1];
        size_t output_d = getOutputParams().dims[2];

        size_t i = 0, j = 0, k = 0;

        for (k = 0; k < input_d; k++)
        {
            for (i = 0; i < input_h; i+=2)
            {
                for (j = 0; j < input_w; j+=2)
                {
                    getOutputData().get<i8>(get_out_flat_idx(i/2, j/2, k, output_w, output_d)) = std::max
                    ({
                        dataIn.get<i8>(get_input_flat_idx(i, j, k, input_w, input_d)),
                        dataIn.get<i8>(get_input_flat_idx(i, j+1, k, input_w, input_d)),
                        dataIn.get<i8>(get_input_flat_idx(i+1, j, k, input_w, input_d)),
                        dataIn.get<i8>(get_input_flat_idx(i+1, j+1, k, input_w, input_d))
                    });
                }
                
            }
            
        }
        
    }

    void MaxPoolingLayer::computeAccelerated(const LayerData &dataIn, const int layer_num) const
    {
        // No mac operation, so just call computeNaive
        computeNaive(dataIn, layer_num);
    }

    void MaxPoolingLayer::computeThreaded(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
    }

    void MaxPoolingLayer::computeTiled(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
    }

    void MaxPoolingLayer::computeSIMD(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
    }
} // namespace ML
