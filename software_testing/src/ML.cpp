#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <filesystem>

#include "Config.h"
#include "Model.h"
#include "Types.h"
#include "Utils.h"
#include "layers/Convolutional.h"
#include "layers/Dense.h"
#include "layers/Layer.h"
#include "layers/MaxPooling.h"
#include "layers/Flatten.h"
#include "layers/Softmax.h"
#include "layers/config.h"

#ifdef ZEDBOARD
#include <file_transfer/file_transfer.h>
//#include "../zedboard/file_transfer/file_transfer.h"
#endif

namespace ML
{

    // Build our ML toy model
    Model buildToyModel(const Path modelPath)
    {
        Model model;
        logInfo("--- Building Toy Model ---");

        // --- Conv 1: L1 ---
        // Input shape: 64x64x3
        // Output shape: 60x60x32
        model.addLayer<ConvolutionalLayer>(
            LayerParams{sizeof(i8), {64, 64, 3}},                                    // Input Data
            LayerParams{sizeof(i8), {60, 60, 32}},                                   // Output Data
            LayerParams{sizeof(i8), {5, 3, 3, 32}, modelPath / "quantized_model_data"/ weights_dir/ "conv2d_weights.bin"}, // Weights
            LayerParams{sizeof(i32), {32}, modelPath / "quantized_model_data"/ biases_dir / "conv2d_biases.bin"}            // Bias
        );

        // --- Conv 2: L2 ---
        // Input shape: 60x60x32
        // Output shape: 56x56x32
        model.addLayer<ConvolutionalLayer>(
            LayerParams{sizeof(i8), {60, 60, 32}},                                    // Input Data
            LayerParams{sizeof(i8), {56, 56, 32}},                                   // Output Data
            LayerParams{sizeof(i8), {5, 3, 32, 32}, modelPath / "quantized_model_data"/weights_dir / "conv2d_1_weights.bin"}, // Weights
            LayerParams{sizeof(i32), {32}, modelPath / "quantized_model_data"/ biases_dir / "conv2d_1_biases.bin"}            // Bias
        );

        // --- MPL 1: L3 ---
        // Input shape: 56x56x32
        // Output shape: 28x28x32
        model.addLayer<MaxPoolingLayer>(
            LayerParams{sizeof(i8), {56, 56, 32}},                                    // Input Data
            LayerParams{sizeof(i8), {28, 28, 32}}                                    // Output Data
        );

        // --- Conv 3: L4 ---
        // Input shape: 28x28x32
        // Output shape: 26x26x64
        model.addLayer<ConvolutionalLayer>(
            LayerParams{sizeof(i8), {28, 28, 32}},                                    // Input Data
            LayerParams{sizeof(i8), {26, 26, 64}},                                   // Output Data
            LayerParams{sizeof(i8), {3, 2, 32, 64}, modelPath / "quantized_model_data"/ weights_dir / "conv2d_2_weights.bin"}, // Weights
            LayerParams{sizeof(i32), {64}, modelPath / "quantized_model_data"/ biases_dir / "conv2d_2_biases.bin"}            // Bias
        );

        // --- Conv 4: L5 ---
        // Input shape: 26x26x64
        // Output shape: 24x24x64
        model.addLayer<ConvolutionalLayer>(
            LayerParams{sizeof(i8), {26, 26, 64}},                                    // Input Data
            LayerParams{sizeof(i8), {24, 24, 64}},                                   // Output Data
            LayerParams{sizeof(i8), {3, 2, 64, 64}, modelPath / "quantized_model_data"/ weights_dir / "conv2d_3_weights.bin"}, // Weights
            LayerParams{sizeof(i32), {64}, modelPath / "quantized_model_data"/ biases_dir / "conv2d_3_biases.bin"}            // Bias
        );

        // --- MPL 2: L6 ---
        // Input shape: 24x24x64
        // Output shape: 12x12x64
        model.addLayer<MaxPoolingLayer>(
            LayerParams{sizeof(i8), {24, 24, 64}},                                    // Input Data
            LayerParams{sizeof(i8), {12, 12, 64}}                                    // Output Data
        );

        // --- Conv 5: L7 ---
        // Input shape: 12x12x64
        // Output shape: 10x10x64
        model.addLayer<ConvolutionalLayer>(
            LayerParams{sizeof(i8), {12, 12, 64}},                                    // Input Data
            LayerParams{sizeof(i8), {10, 10, 64}},                                   // Output Data
            LayerParams{sizeof(i8), {3, 2, 64, 64}, modelPath / "quantized_model_data"/ weights_dir / "conv2d_4_weights.bin"}, // Weights
            LayerParams{sizeof(i32), {64}, modelPath / "quantized_model_data"/ biases_dir / "conv2d_4_biases.bin"}            // Bias
        );

        // --- Conv 6: L8 ---
        // Input shape: 10x10x64
        // Output shape: 8x8x128
        model.addLayer<ConvolutionalLayer>(
            LayerParams{sizeof(i8), {10, 10, 64}},                                    // Input Data
            LayerParams{sizeof(i8), {8, 8, 128}},                                   // Output Data
            LayerParams{sizeof(i8), {3, 2, 64, 128}, modelPath / "quantized_model_data"/ weights_dir / "conv2d_5_weights.bin"}, // Weights
            LayerParams{sizeof(i32), {128}, modelPath / "quantized_model_data"/ biases_dir / "conv2d_5_biases.bin"}            // Bias
        );

        // --- MPL 3: L9 ---
        // Input shape: 8x8x128
        // Output shape: 4x4x128
        model.addLayer<MaxPoolingLayer>(
            LayerParams{sizeof(i8), {8, 8, 128}},                                    // Input Data
            LayerParams{sizeof(i8), {4, 4, 128}}                                    // Output Data
        );

        // --- Flatten 1: L10 ---
        // Input shape: 4x4x128
        // Output shape: 2048
        model.addLayer<FlattenLayer>(
            LayerParams{sizeof(i8), {4, 4, 128}},                                    // Input Data
            LayerParams{sizeof(i8), {2048}}                                         // Output Data
        );

        // --- Dense 1: L11 ---
        // Input shape: 2048
        // Output shape: 256
        model.addLayer<DenseLayer>(
            LayerParams{sizeof(i8), {2048}},                                    // Input Data
            LayerParams{sizeof(i8), {256}},                                   // Output Data
            LayerParams{sizeof(i8), {256, 1024}, modelPath / "quantized_model_data"/ weights_dir / "dense_weights.bin"}, // Weights
            LayerParams{sizeof(i32), {256}, modelPath / "quantized_model_data"/ biases_dir/ "dense_biases.bin"}  ,    // Bias
            DenseLayer::ActivationType::ReLU
        );

        // --- Dense 2: L12 ---
        // Input shape: 256
        // Output shape: 200
        model.addLayer<DenseLayer>(
            LayerParams{sizeof(i8), {256}},                                    // Input Data
            LayerParams{sizeof(fp32), {200}},                                   // Output Data
            LayerParams{sizeof(i8), {200, 128}, modelPath / "quantized_model_data"/ weights_dir / "dense_1_weights.bin"}, // Weights
            LayerParams{sizeof(i32), {200}, modelPath / "quantized_model_data"/ biases_dir / "dense_1_biases.bin"}  ,    // Bias
            DenseLayer::ActivationType::SoftMax

        );

        // --- Softmax 1: L13 ---
        // Input shape: 200
        // Output shape: 200

        // I added softmax as an activation funciton to the previous one. So no need to implement it. I don't know why they have it here
        // as a separate layer, since ReLU activation function was always done before the output of a layer

        return model;
    }

    inline size_t get_image_flat_idx(size_t height_idx, size_t width_idx, size_t depth_idx, size_t image_w, size_t image_d){
        return image_w * image_d * height_idx + image_d * width_idx + depth_idx;
    }

    void runInferenceTest(const Model &model, const Path& input_image_bin, const Path& ground_truth_pred_path)
    {
        // Load an image
        logInfo("\n\n\n--- Running Inference Test ---");

        // read the original file into buffer, the size of the input is fp32
        // there is no way of reading the image and quantizing it on the go, 
        // so first we must read it as it is, and then quantize it by writing to a different buffer
        LayerData unquantized_img({sizeof(fp32), model[0].getInputParams().dims}, input_image_bin);
        unquantized_img.loadData();

        LayerData quantized_image(model[0].getInputParams());
        quantized_image.allocData();

        for (size_t i = 0; i < model[0].getInputParams().flat_count(); i++)
        {
            i8 temp = static_cast<i8>(std::clamp(std::nearbyint((SI_VALS[0] * unquantized_img.get<fp32>(i) + SZ_VALS[0])), min_value, max_value));

            quantized_image.get<i8>(i) = temp;
        }

        // free the unquantized image, we don't need to hold in our memory anymore
        unquantized_img.freeData();

        Timer timer("Full Inference");

        // Run inference on the model
        timer.start();
        const LayerData &output = model.inference(quantized_image, Layer::InfType::NAIVE);
        timer.stop();

        // Compare the output
        // Construct a LayerData object from a LayerParams one
        LayerData expected(model.getOutputLayer().getOutputParams(), ground_truth_pred_path);
        expected.loadData();
        output.compareWithinPrint<fp32>(expected);
    }

    void runTests()
    {
        // Base input data path (determined from current directory of where you are running the command)
        Path basePath("./data"); // May need to be altered for zedboards loading from SD Cards

        // Build the model and allocate the buffers
        Model model = buildToyModel(basePath);
        model.allocLayers(); // ME: allocates memory to store out_data from each leayer

        runInferenceTest(model, basePath / "test_input"/"test_input_image.bin", basePath / "test_input_feature_maps" / "dense_1_feature.bin");
        //runInferenceTest(model, basePath / "test_input_feature_maps"/"dense_feature.bin", basePath / "test_input_feature_maps" / "dense_1_feature.bin");

        // Clean up
        model.freeLayers();
        std::cout << "\n\n----- ML::runTests() COMPLETE -----\n";
    }

} // namespace ML

#ifdef ZEDBOARD
extern "C" int main()
{
    try
    {
        static FATFS fatfs;
        if (f_mount(&fatfs, "/", 1) != FR_OK)
        {
            throw std::runtime_error("Failed to mount SD card. Is it plugged in?");
        }
        ML::runTests();
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n\n----- EXCEPTION THROWN -----\n"
                  << e.what() << '\n';
    }
    std::cout << "\n\n----- STARTING FILE TRANSFER SERVER -----\n";
    FileServer::start_file_transfer_server();
}
#else
int main()
{
    ML::runTests();
}
#endif
