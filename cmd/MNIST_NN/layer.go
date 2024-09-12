package main

import (
	"math"
	"math/rand"
)



type Layer struct {
	NumIn int
	NumOut int

	Weights []float64
	Biases  []float64


	lossGradientW, lossGradientB []float64

	Activations []float64
	ActivationFunc Activation

	weightedInputs []float64
}


func NewLayer(numIn, numOut int ,randomizer *rand.Rand) *Layer{
	l := &Layer{
		NumIn: numIn,
		NumOut: numOut,
	}

	l.Weights = make([]float64, numIn*numOut)
	l.Biases = make([]float64, numOut)

	l.lossGradientW = make([]float64, numOut*numIn)
	l.lossGradientB = make([]float64, numOut)

	l.Activations = make([]float64, numOut)
	l.weightedInputs = make([]float64, numOut)

	l.InitializeRandomWeights(randomizer)
	l.InitializeRandomBiases(randomizer)


	return l
}

func (l *Layer) SetActivationFn(a ActivationType){
	l.ActivationFunc = GetActivationFunc(a)
}

func (l *Layer) InitializeRandomWeights(randomizer *rand.Rand){
	for i := range l.Weights{
		l.Weights[i] = randomIn(randomizer, 1) / math.Sqrt((float64(l.NumIn)))
	}
}

func (l *Layer) InitializeRandomBiases(randomizer *rand.Rand){
	for i := range l.Biases{
		l.Biases[i] = randomIn(randomizer, 1) / math.Sqrt((float64(l.NumIn)))
	}
}

func randomIn(randomizer *rand.Rand, lambda float64) float64{
	u := randomizer.Float64()
	return -math.Log(1-u) / lambda
}

func (l *Layer) CalculatingLearningOutputs(inputs []float64) []float64{

	for nodeOut := 0; nodeOut< l.NumOut; nodeOut++{
		weightedIn := l.Biases[nodeOut]
		for nodeIn := 0; nodeIn< l.NumIn; nodeIn++{
			weightedIn += inputs[nodeIn] * l.getWeight(nodeIn, nodeOut)
		}
		l.weightedInputs[nodeOut] = weightedIn
	}
	for i := range l.Activations{
		l.Activations[i] = l.ActivationFunc.function(l.weightedInputs, i)
	}

	return l.Activations
}

func (l *Layer) getWeight(nodeIn, nodeOut int) float64{
	return l.Weights[nodeOut * l.NumIn + nodeIn]
}



func (l *Layer) CalculateLastLayerLossGradientbyW(expectedOutputs []float64, loss Loss) {
    numOutputs := len(expectedOutputs)

    for i := 0; i < numOutputs; i++ {

        lossDerivative := loss.prime_function(l.Activations[i], expectedOutputs[i])

        for j := 0; j < l.NumIn; j++ {
            weightIndex := i * l.NumIn + j

            activationDerivative := l.ActivationFunc.prime_function(l.Weights, j)
            l.lossGradientW[weightIndex] = lossDerivative * activationDerivative
        }
    }
}



func (l *Layer) CalculateHiddenLayerLossGradientbyW(nextLayer *Layer, oldNodeValues []float64) {
    numInputs := len(oldNodeValues)

    for i := 0; i < l.NumOut; i++ {
        var errorFromNextLayer float64
        for j := 0; j < nextLayer.NumOut; j++ {
            errorFromNextLayer += nextLayer.lossGradientW[j * nextLayer.NumIn + i] * nextLayer.getWeight(i, j)
        }
        
        activationDerivative := l.ActivationFunc.prime_function(l.weightedInputs, i)
        
        for j := 0; j < numInputs; j++ {
            weightIndex := i * numInputs + j
            l.lossGradientW[weightIndex] = errorFromNextLayer * activationDerivative * oldNodeValues[j]
        }
    }
}


func (l *Layer) UpdateWeights(learningRate float64) {
	for i := range l.Weights {
		l.Weights[i] -= learningRate * l.lossGradientW[i]
	}
	for i := range l.Biases {
		l.Biases[i] -= learningRate * l.lossGradientB[i]
	}
}


		
