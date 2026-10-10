#include "Dense.h"

#include <iostream>
#include <cmath>
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

    // Compute the convultion for the layer data
    // ME: the LayerData object passed has been loaded before (by img.loadData() no need to do it again)
    void DenseLayer::computeNaive(const LayerData &dataIn, const int layer_num) const
    {
        /*
            Here is how the dense layer works:
            Although we could have implemented the dense layer as a special case of convolution by fixing the kernel size
            to the input feature map size, I don't think it is commonly done that way
            Apparently there usually sits a flatten layer in between.
            Flatten layer outputs a single 1D vector
            In our case flatten layer's output shape is <2048> and our dense layer's kernel size is <2048 x 256>

            that means the output from out dense layer has to be of shape <256> 1D array

            The math is as following:
            Since the weights are of shape 2048 x 256 the best way to think about it is this:
            fix the column of the matrix to 0, and iterate over the rows. (go down) those 2048 numbers correspond
            to the weights from all 2048 neurons of input data to the fist neuron of the output data
        */
        size_t input_neuron_count  = getInputParams().dims[0];  // 2048
        size_t output_neuron_count = getOutputParams().dims[0]; // 256

        std::vector<i32> output_before_activation;

        size_t i = 0, j = 0;
        i32 sum = 0;

        // dense_weights_2D is a pointer that points to a collection of 256 elemetns that store fp32
        // type cast the 1d array into pointer to an array of 256 floats and assign it to dense_weights_2D
//        const fp32 (*dense_weights_2D)[output_neuron_count] = (fp32 (*)[output_neuron_count])(getWeightData().raw());

     
        for (i = 0; i < output_neuron_count; i++){
            for (j = 0; j < input_neuron_count; j++){

                sum+= getWeightData().get<i8>(output_neuron_count * j + i) * dataIn.get<i8>(j);
            }

            i32 sum_plus_bias = sum + getBiasData().get<i32>(i);

            output_before_activation.push_back(sum_plus_bias);

            sum = 0;
        }
        fp32 softmax_denominator_val = 0;

        switch (get_activation_type())
        {
        case ActivationType::SoftMax:

            for (size_t i = 0; i < output_neuron_count; i++)
            {
                softmax_denominator_val += std::exp(output_before_activation[i]/(SI_VALS[layer_num] * SW_VALS[layer_num]));
            }
            for (i = 0; i < output_neuron_count; i++)
            {
                getOutputData().get<fp32>(i) = std::exp(output_before_activation[i]/(SI_VALS[layer_num] * SW_VALS[layer_num])) / softmax_denominator_val;
            }
            break;



        case ActivationType::ReLU:
        {
            float M      = SI_VALS[layer_num + 1] / (SI_VALS[layer_num] * SW_VALS[layer_num]);
            float z_next = SZ_VALS[layer_num + 1];
            for (i = 0; i < output_neuron_count; i++)
            {
                
                float v = std::nearbyint(output_before_activation[i] * M) + z_next;   // requantize
                v = std::clamp(v, z_next, max_value);                                        // ReLU + saturation
                getOutputData().get<i8>(i) = static_cast<i8>(v);
            }
          break;
        }
        
        default:
            break;
        }
    }

    void DenseLayer::computeAccelerated(const LayerData &dataIn, const int layer_num) const
    {
//        computeNaive(dataIn, layer_num);
        size_t input_neuron_count  = getInputParams().dims[0];  // 2048
        size_t output_neuron_count = getOutputParams().dims[0]; // 256

        std::vector<i32> output_before_activation;

        size_t i = 0, j = 0;

        // dense_weights_2D is a pointer that points to a collection of 256 elemetns that store fp32
        // type cast the 1d array into pointer to an array of 256 floats and assign it to dense_weights_2D
//        const fp32 (*dense_weights_2D)[output_neuron_count] = (fp32 (*)[output_neuron_count])(getWeightData().raw());

     
        for (i = 0; i < output_neuron_count; i++){
            
            i32 bias = getBiasData().get<i32>(i);

            // load bias as signed 32 bit
            Xil_Out32(XPAR_AXI_FIFO_0_BASEADDR + XLLF_TDFD_OFFSET, bias); 

            for (j = 0; j < input_neuron_count; j++){

                i8 weight = getWeightData().get<i8>(output_neuron_count * j + i);
                i8 activation = dataIn.get<i8>(j);

                // send these two by concatinating them
                Xil_Out32(XPAR_AXI_FIFO_0_BASEADDR + XLLF_TDFD_OFFSET, ML::concat_hex_to_i32(activation, weight)); 
            }

            // send TLAST and how many data we sent
            // we sent (j) + one_bias    each data that we sent was 4 bytes
            Xil_Out32(XPAR_AXI_FIFO_0_BASEADDR + XLLF_TLF_OFFSET, 4 * (j + 1));

            i32 mac_out = ML::read_packet();

            output_before_activation.push_back(mac_out);

        }
        fp32 softmax_denominator_val = 0;

        switch (get_activation_type())
        {
        case ActivationType::SoftMax:

            for (size_t i = 0; i < output_neuron_count; i++)
            {
                softmax_denominator_val += std::exp(output_before_activation[i]/(SI_VALS[layer_num] * SW_VALS[layer_num]));
            }
            for (i = 0; i < output_neuron_count; i++)
            {
                getOutputData().get<fp32>(i) = std::exp(output_before_activation[i]/(SI_VALS[layer_num] * SW_VALS[layer_num])) / softmax_denominator_val;
            }
            break;



        case ActivationType::ReLU:
        {
            float M      = SI_VALS[layer_num + 1] / (SI_VALS[layer_num] * SW_VALS[layer_num]);
            float z_next = SZ_VALS[layer_num + 1];
            for (i = 0; i < output_neuron_count; i++)
            {
                
                float v = std::nearbyint(output_before_activation[i] * M) + z_next;   // requantize
                v = std::clamp(v, z_next, max_value);                                        // ReLU + saturation
                getOutputData().get<i8>(i) = static_cast<i8>(v);
            }
          break;
        }
        
        default:
            break;
        }
    }

    void DenseLayer::computeThreaded(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
    }

    void DenseLayer::computeTiled(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
    }

    void DenseLayer::computeSIMD(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
    }
} // namespace ML
