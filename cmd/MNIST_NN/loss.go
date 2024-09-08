package main

//import "math"


type LossType int 



const(
	MSE_T LossType = iota
	CrossEntropy_T
)


type Loss interface{
	function(predictedOutputs,expectedOutputs []float64) float64
	prime_function(predictedOutputs,expectedOutputs float64) float64
}
type loss struct{}

func GetLossFunc(lossType LossType) Loss{
	switch lossType {
	case MSE_T:
		return MeanSquaredError{}
	
	case CrossEntropy_T:
		return nil/*CrossEntropy{}*/
	
	
	default:
		panic("Unavailable activation type for the moment")
	}
}

type MeanSquaredError struct{}

func (mse MeanSquaredError) function(predictedOutputs, expectedOutputs []float64) float64{
	
	var loss float64
	for i:= 0; i<len(predictedOutputs); i++{
		error := predictedOutputs[i] - expectedOutputs[i]
		loss += error * error
	}
	return loss * .5
	
}

func (mse MeanSquaredError) prime_function(predictedOutputs, expectedOutputs float64) float64{
	return predictedOutputs - expectedOutputs
}



type CrossEntropy struct{}

/*func (ce CrossEntropy) function(predictedOutputs, expectedOutputs []float64) float64  {
	const epsilon = 1e-15

	for i := 0; i< len(predictedOutputs); i++{
		predictedOutputs[i] = math.Max(math.Min(predictedOutputs[i],1-epsilon),epsilon)
		sum += expectedOutputs[i]*math.Log(predictedOutputs[i]) + (1 - predictedOutputs[i])*math.Log(1-predictedOutputs[i])
	}
} */



