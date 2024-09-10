package main

import (
	"math/rand"
	"time"
)

func main() {
	
	randomizer := rand.New(rand.NewSource(time.Now().UnixNano()))

	
	nn := NewNN(0.01, 50, ReLU, MSE_T)

	
	nn.AddLayer(3, 4, randomizer) 
	nn.AddLayer(4, 2, randomizer) 

	
	inputs := [][]float64{
		{0.5, 0.1, 0.3},
	}
	expectedOutputs := [][]float64{
		{1, 0},

	}


	nn.Train(inputs, expectedOutputs)

	

}
