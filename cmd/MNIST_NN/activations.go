package main

import (
	"math"
)


type ActivationType int


const (
	Sigmoid ActivationType = iota
	ReLU
	Softmax
)

type Activation interface {
	function(inputs []float64, index int) float64
	prime_function(inputs []float64, index int) float64

}


func GetActivationFunc(activationType ActivationType) Activation{
	switch activationType {
	case Sigmoid:
		return SigmoidActivation{}
	
	case ReLU:
		return ReLUActivation{}
	
	case Softmax:
		return SoftmaxActivation{}
	
	default:
		panic("Unavailable activation type for the moment")
	}
}


type SigmoidActivation struct{}

func (a SigmoidActivation) function(inputs []float64, index int) float64 {
	/*
	Sigmoid(x) = 1 / (1 + e^(-x))
	*/
	return 1 / (1 + math.Exp(- inputs[index]))
}

func (a SigmoidActivation) prime_function(inputs []float64, index int) float64{
	sig := a.function(inputs,index)
	return sig * (1 - sig)
}

type ReLUActivation struct{}

func (a ReLUActivation) function(inputs []float64, index int) float64 {
	/*
	ReLU(x) = max(0, x)
	*/
	return math.Max(0, inputs[index])
}

func (a ReLUActivation) prime_function(inputs []float64, index int) float64{
	if inputs[index] > 0 {
		return 1
	}
	return 0
}

type SoftmaxActivation struct{}

func (a SoftmaxActivation) function(inputs []float64, index int) float64{
	/*
	Softmax(x_i) = e^(x_i) / sum(e^(x_j) for j in 1 to n)
	*/
	sum := .0
	for _, input := range inputs{
		sum += math.Exp(input)
	}
	return math.Exp(inputs[index]) / sum
} 

func (a SoftmaxActivation) prime_function(inputs []float64, index int) float64 {
	sum := .0
	for _, input := range inputs{
		sum += math.Exp(input)
	}
	x := math.Exp(inputs[index]) / sum

	return (x * sum - x * x) / sum * sum
	
}



