package main

import (
    "encoding/json"
    "os"
)

// LoadModel loads a neural network model from a JSON file.
func LoadModel(filename string) (*NN, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    var modelData map[string]interface{}
    decoder := json.NewDecoder(file)
    if err := decoder.Decode(&modelData); err != nil {
        return nil, err
    }

    nn := &NN{
        learningRate: modelData["learningRate"].(float64),
        epochs:       uint64(modelData["epochs"].(float64)),
        ActivationFn: parseActivationType(modelData["activationFn"].(string)),
        LossFn:       parseLossType(modelData["lossFn"].(string)),
    }

    layerData := modelData["layers"].([]interface{})
    for _, lData := range layerData {
        layerMap := lData.(map[string]interface{})
        numIn := int(layerMap["numIn"].(float64))
        numOut := int(layerMap["numOut"].(float64))

        layer := NewLayer(numIn, numOut, nil) 
        layer.Weights = layerMap["weights"].([]float64)
        layer.Biases = layerMap["biases"].([]float64)

        layer.SetActivationFn(nn.ActivationFn)
        nn.layers = append(nn.layers, layer)
    }

    return nn, nil
}

func parseActivationType(s string) ActivationType {
    switch s {
    case "sigmoid":
        return Sigmoid
    case "relu":
        return ReLU
    default:
        return Sigmoid 
    }
}

func parseLossType(s string) LossType {
    switch s {
    case "cross_entropy":
        return CrossEntropy_T
    case "mse":
        return MSE_T
    default:
        return CrossEntropy_T
    }
}


func GetMaxIndex(arr []float64) int {
    maxIndex := 0
    for i, value := range arr {
        if value > arr[maxIndex] {
            maxIndex = i
        }
    }
    return maxIndex
}