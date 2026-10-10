#include "Convolutional.h"

#include <iostream>
#include <algorithm>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"
#include "./config.h"

#ifdef ZEDBOARD
    #include "fifo_transaction.h"
#endif

namespace ML
{
    // --- Begin Student Code ---

    inline size_t get_kernel_flat_idx(size_t height_idx, size_t width_idx, size_t depth_idx, size_t batch_idx, size_t kernel_w, size_t kernel_d, size_t kernel_b){
        return kernel_w * kernel_d * kernel_b * height_idx + kernel_d * kernel_b * width_idx + kernel_b * depth_idx + batch_idx;
    }

    inline size_t get_image_flat_idx(size_t height_idx, size_t width_idx, size_t depth_idx, size_t image_w, size_t image_d){
        return image_w * image_d * height_idx + image_d * width_idx + depth_idx;
    }

    inline size_t get_out_flat_idx(size_t height_idx, size_t width_idx, size_t batch_idx, size_t out_w, size_t out_d){
        return out_w * out_d * height_idx + out_d * width_idx + batch_idx;
    }

    // Compute the convultion for the layer data
    // ME: the LayerData object passed has been loaded before (by img.loadData() no need to do it again)
    void ConvolutionalLayer::computeNaive(const LayerData &dataIn, const int layer_num) const
    {
        // TODO: Your Code Here...
        // The following line is an example of copying a single 32-bit floating point integer from the input layer data to the output layer data
        // ME: so instead of doing this, I have to put the correct one to the output (The convolved version of the input)
        
        //std::cout << "bias count of the conv layer " << getBiasData().getParams().flat_count() << std::endl;
        //std::cout << "fist 10 weights: \n" << std::endl;
        //for (size_t i = 0; i < 10; i++)
        //{
        //    std::cout << i << ": " << getWeightData().get<fp32>(i) << std::endl;
        //}

        //getOutputData().get<fp32>(0) = dataIn.get<fp32>(0);

        // {64, 64, 3}
        size_t image_w = getInputParams().dims[1];
        size_t image_d = getInputParams().dims[2];

        // {60, 60, 32}
        size_t out_h = getOutputParams().dims[0];
        size_t out_w = getOutputParams().dims[1];

        // {5, 5, 3, 32}
        size_t kernel_h = getWeightParams().dims[0];
        size_t kernel_w = getWeightParams().dims[1];
        size_t kernel_d = getWeightParams().dims[2];
        size_t kernel_b = getWeightParams().dims[3];

    
        size_t j = 0, l = 0, i = 0, k = 0, d = 0, b = 0;
    
        // where I accumulate the sum

        fp32 M;      
        fp32 z_next; 
        
        // there should be a smarter way of doing this, but basically the idea is to choose the correct scale
        // if because layer_num + 1 might be 0
        if (layer_num == 1 || layer_num == 4 || layer_num == 5){
            M      = SI_VALS[layer_num + 2] / (SI_VALS[layer_num] * SW_VALS[layer_num]);
            z_next = SZ_VALS[layer_num + 2];
        }
        else if (layer_num == 7){
            M      = SI_VALS[layer_num + 3] / (SI_VALS[layer_num] * SW_VALS[layer_num]);
            z_next = SZ_VALS[layer_num + 3];
        }
        else {
            M      = SI_VALS[layer_num + 1] / (SI_VALS[layer_num] * SW_VALS[layer_num]);
            z_next = SZ_VALS[layer_num + 1];
        }

        i32 sum;

        for (b = 0; b < kernel_b; b++){
            for (j = 0; j < out_h; j++){
                for (l = 0; l < out_w; l++){

                    sum = getBiasData().get<i32>(b);

                    for (d = 0; d < kernel_d; d++){
                        for (i = 0; i < kernel_h; i++){
                            for (k = 0; k < kernel_w; k++){
                                sum += 
                                (
                                    getWeightData().get<i8>(get_kernel_flat_idx(i, k, d, b, kernel_w, kernel_d, kernel_b))
                                    *
                                    dataIn.get<i8>(get_image_flat_idx(j + i, l + k, d, image_w, image_d))
                                );
                            }
                        }
                    }

                    fp32 out_val = std::round(sum * M) + z_next;   // requantize
                    out_val = std::clamp(out_val, z_next, max_value);                    // RELU 
                    
                    // cast it to i8 and assign to output
                    getOutputData().get<i8>(get_out_flat_idx(j, l, b, out_w, kernel_b)) = static_cast<i8>(out_val);
               }
           }
        }
    }

    // Compute the convolution using Hardware MAC 
    void ConvolutionalLayer::computeAccelerated(const LayerData &dataIn, const int layer_num) const
    {
//        computeNaive(dataIn, layer_num);
        size_t image_w = getInputParams().dims[1];
        size_t image_d = getInputParams().dims[2];

        // {60, 60, 32}
        size_t out_h = getOutputParams().dims[0];
        size_t out_w = getOutputParams().dims[1];

        // {5, 5, 3, 32}
        size_t kernel_h = getWeightParams().dims[0];
        size_t kernel_w = getWeightParams().dims[1];
        size_t kernel_d = getWeightParams().dims[2];
        size_t kernel_b = getWeightParams().dims[3];

    
        size_t j = 0, l = 0, i = 0, k = 0, d = 0, b = 0;
    
        // where I accumulate the sum

        fp32 M;      
        fp32 z_next; 
        
        // there should be a smarter way of doing this, but basically the idea is to choose the correct scale
        // if because layer_num + 1 might be 0
        if (layer_num == 1 || layer_num == 4 || layer_num == 5){
            M      = SI_VALS[layer_num + 2] / (SI_VALS[layer_num] * SW_VALS[layer_num]);
            z_next = SZ_VALS[layer_num + 2];
        }
        else if (layer_num == 7){
            M      = SI_VALS[layer_num + 3] / (SI_VALS[layer_num] * SW_VALS[layer_num]);
            z_next = SZ_VALS[layer_num + 3];
        }
        else {
            M      = SI_VALS[layer_num + 1] / (SI_VALS[layer_num] * SW_VALS[layer_num]);
            z_next = SZ_VALS[layer_num + 1];
        }

        for (b = 0; b < kernel_b; b++){
            for (j = 0; j < out_h; j++){
                for (l = 0; l < out_w; l++){

                    i32 bias = getBiasData().get<i32>(b);
                    // load bias as signed 32 bit
                    Xil_Out32(XPAR_AXI_FIFO_0_BASEADDR + XLLF_TDFD_OFFSET, bias); 
 
                    for (d = 0; d < kernel_d; d++){
                        for (i = 0; i < kernel_h; i++){
                            for (k = 0; k < kernel_w; k++){
                                i8 weight = getWeightData().get<i8>(get_kernel_flat_idx(i, k, d, b, kernel_w, kernel_d, kernel_b));
                                i8 activation = dataIn.get<i8>(get_image_flat_idx(j + i, l + k, d, image_w, image_d));

                                // send these two by concatinating them
                                Xil_Out32(XPAR_AXI_FIFO_0_BASEADDR + XLLF_TDFD_OFFSET, ML::concat_hex_to_i32(activation, weight)); 
                            }
                        }
                    }
                    // send TLAST and how many data we sent
                    // we sent (d * i * k) + one_bias    each data that we sent was 4 bytes
                    Xil_Out32(XPAR_AXI_FIFO_0_BASEADDR + XLLF_TLF_OFFSET, 4 * (d * i * k + 1));

                    i32 mac_out = ML::read_packet();

                    fp32 out_val = std::round(mac_out * M) + z_next;   // requantize
                    out_val = std::clamp(out_val, z_next, max_value);                    // RELU 
                    
                    // cast it to i8 and assign to output
                    getOutputData().get<i8>(get_out_flat_idx(j, l, b, out_w, kernel_b)) = static_cast<i8>(out_val);
               }
           }
        }
    }


    // Compute the convolution using threads
    void ConvolutionalLayer::computeThreaded(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
    }

    // Compute the convolution using a tiled approach
    void ConvolutionalLayer::computeTiled(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
    }

    // Compute the convolution using SIMD
    void ConvolutionalLayer::computeSIMD(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
    }
} // namespace ML
