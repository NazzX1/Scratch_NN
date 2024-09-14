package main

import (
	"math/rand"
	"time"
)

func main() {
	
	data := LoadDataset("../../data/mnist_train.csv")

	images, labels := PrepareDataset(data)

	images, labels = ShuffleData(images, labels)

	encodedLabels := ToOneHot(labels, 10)

	imagesBatch, labelsBatches:= CreateBatches(images, encodedLabels, 128)


	randomizer := rand.New(rand.NewSource(time.Now().UnixNano()))
	nn := NewNN(0.01, 10, ReLU, MSE_T) 

	nn.AddLayer(784, 128, randomizer)
	nn.AddLayer(128, 10, randomizer)

	nn.Train(imagesBatch, labelsBatches)




	

}
