package main

import (
	"fmt"
	"math/rand"
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

func (nn *NN) AddLayer(numIn, numOut int, randomizer *rand.Rand){
	layer := NewLayer(numIn, numOut,randomizer)
	layer.SetActivationFn(nn.ActivationFn)
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
		// Iterate over each sample
		for i := 0; i < numSamples; i++ {
			// Get the input and expected output for the current sample
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
    totalPredictions := len(inputs) // Set totalPredictions to the number of input samples

    for i, input := range inputs {
        // Forward pass
        predictions := nn.ForwardPropagation(input)

        // Get the predicted class (index of the highest value)
        predictedClass := GetMaxIndex(predictions)

        // Get the true class from the one-hot encoded target output
        trueClass := GetMaxIndex(expectedOutputs[i]) // Use expectedOutputs instead of inputs

        // Compare prediction to true class
        if predictedClass == trueClass {
            correctPredictions++
        }
    }

    // Calculate accuracy
    accuracy := float64(correctPredictions) / float64(totalPredictions)
    return accuracy
}




