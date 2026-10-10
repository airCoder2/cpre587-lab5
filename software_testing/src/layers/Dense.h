#pragma once

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"
#include "./config.h"

namespace ML {
class DenseLayer : public Layer {
   public:
    enum class ActivationType {ReLU, SoftMax};

    DenseLayer(const LayerParams inParams, const LayerParams outParams, const LayerParams weightParams, const LayerParams biasParams, const ActivationType activation_type)
        : Layer(inParams, outParams, LayerType::DENSE),
          weightParam(weightParams), 
          weightData(weightParams),
          biasParam(biasParams),
          biasData(biasParams),
          activation_type(activation_type){}



    // Getters
    const LayerParams& getWeightParams() const { return weightParam; }
    const LayerParams& getBiasParams() const { return biasParam; }
    const LayerData& getWeightData() const { return weightData; }
    const LayerData& getBiasData() const { return biasData; }
    const ActivationType& get_activation_type() const {return activation_type;}

    // Allocate all resources needed for the layer & Load all of the required data for the layer
    virtual void allocLayer() override {
        Layer::allocLayer();
        weightData.loadData();
        biasData.loadData();
    }

    // Fre all resources allocated for the layer
    virtual void freeLayer() override {
        Layer::freeLayer();
        weightData.freeData();
        biasData.freeData();
    }

    // Virtual functions
    virtual void computeAccelerated(const LayerData &dataIn, const int layer_num) const override;
    virtual void computeNaive(const LayerData& dataIn, const int layer_num) const override;
    virtual void computeThreaded(const LayerData& dataIn) const override;
    virtual void computeTiled(const LayerData& dataIn) const override;
    virtual void computeSIMD(const LayerData& dataIn) const override;

   private:
    LayerParams weightParam;
    LayerData weightData;

    LayerParams biasParam;
    LayerData biasData;

    ActivationType activation_type;
};

}  // namespace ML