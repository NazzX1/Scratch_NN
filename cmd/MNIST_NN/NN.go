package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
)

type NN struct {
	layers       []*Layer
	learningRate float64
	epochs       uint64

	ActivationFn ActivationType
	LossFn       LossType
}

func NewNN(lr float64, e uint64, activationType ActivationType, lossType LossType) *NN {
	return &NN{
		layers:       nil,
		learningRate: lr,
		epochs:       e,
		ActivationFn: activationType,
		LossFn:       lossType,
	}
}

func (nn *NN) AddLayer(numIn, numOut int, randomizer *rand.Rand, ActivationF ActivationType){
	layer := NewLayer(numIn, numOut,randomizer)
	layer.SetActivationFn(ActivationF)
	nn.layers = append(nn.layers, layer)
}

func (nn *NN) ForwardPropagation(inputs []float64) []float64{
	for _, layer := range nn.layers{
		inputs = layer.CalculatingLearningOutputs(inputs)
	}

	return inputs
}

func (nn *NN) BackwardPropagation(expectedOutputs []float64) {
	outputLayer := nn.layers[len(nn.layers)-1]

	outputLayer.CalculateLastLayerLossGradientbyW(expectedOutputs, GetLossFunc(nn.LossFn))

	for i := len(nn.layers) - 2; i >= 0; i--{
		layer := nn.layers[i]
		nexLayer := nn.layers[i+1]

		layer.CalculateHiddenLayerLossGradientbyW(nexLayer, layer.Activations)
	}
}


func (nn *NN) UpdateWeights(){
	for _, layer := range nn.layers{
		layer.UpdateWeights(nn.learningRate)
	}
}



func (nn *NN) Train(inputs [][]float64, expectedOutputs [][]float64) {
	numSamples := len(inputs)

	for epoch := 0; epoch < int(nn.epochs); epoch++ {
		for i := 0; i < numSamples; i++ {
			input := inputs[i]
			targets := expectedOutputs[i]

			// Forward pass
			activations := nn.ForwardPropagation(input)

			// Compute loss
			loss := GetLossFunc(nn.LossFn).function(activations, targets)
			fmt.Printf("Epoch %d, Sample %d, Loss: %f\n", epoch, i, loss)

			// Backward pass
			nn.BackwardPropagation(targets)

			// Update weights
			nn.UpdateWeights()
		}
	}
}



func (nn *NN) Predict(inputs []float64) []float64 {
	for _, layer := range nn.layers {
		inputs = layer.CalculatingLearningOutputs(inputs)
	}
	return inputs
}


func (nn *NN) CalculateAccuracy(inputs [][]float64, expectedOutputs [][]float64) float64 {
    correctPredictions := 0
    totalPredictions := len(inputs)

    for i, input := range inputs {

        predictions := nn.ForwardPropagation(input)

        predictedClass := GetMaxIndex(predictions)

        trueClass := GetMaxIndex(expectedOutputs[i]) 

        if predictedClass == trueClass {
            correctPredictions++
        }
    }

    accuracy := float64(correctPredictions) / float64(totalPredictions)
    return accuracy
}

func (nn *NN) SaveModel(filename string) error {
	modelData := make(map[string]interface{})

	layerData := make([]map[string]interface{}, len(nn.layers))
	for i, layer := range nn.layers {
		layerData[i] = map[string]interface{}{
			"numIn":     layer.NumIn,
			"numOut":    layer.NumOut,
			"weights":   layer.Weights,
			"biases":    layer.Biases,
		}
	}

	modelData["layers"] = layerData
	modelData["learningRate"] = nn.learningRate
	modelData["epochs"] = nn.epochs
	modelData["activationFn"] = nn.ActivationFn
	modelData["lossFn"] = nn.LossFn

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	return encoder.Encode(modelData)
}




