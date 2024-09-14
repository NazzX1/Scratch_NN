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

		layer.CalculateHiddenLayerLossGradientbyW(nexLayer, nil)
	}
}


func (nn *NN) UpdateWeights(){
	for _, layer := range nn.layers{
		layer.UpdateWeights(nn.learningRate)
	}
}



func (nn *NN) Train(inputs [][][]float64, expectedOutputs [][][]float64) {
	for epoch := 0; epoch < int(nn.epochs); epoch++ {
		for batchIndex, batchInputs := range inputs {
			batchOutputs := expectedOutputs[batchIndex]

			for i, input := range batchInputs {
				// Forward pass
				activations := nn.ForwardPropagation(input)

				// Get the one-hot encoded target output
				targets := batchOutputs[i]

				// Compute loss
				loss := GetLossFunc(nn.LossFn).function(activations, targets)
				fmt.Printf("Epoch %d, Batch %d, Sample %d, Loss: %f\n", epoch, batchIndex, i, loss)

				// Backward pass
				nn.BackwardPropagation(targets)

				// Update weights
				nn.UpdateWeights()
			}
		}
	}
}



func (nn *NN) Predict(inputs []float64) []float64 {
	for _, layer := range nn.layers {
		inputs = layer.CalculatingLearningOutputs(inputs)
	}
	return inputs
}


