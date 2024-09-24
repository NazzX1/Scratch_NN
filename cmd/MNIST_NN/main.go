package main

import (
	"fmt"
	"math/rand"
	"time"
)


func main() {
	// Load and prepare the dataset
	data := LoadDataset("../../data/mnist_train.csv")
	images, labels := PrepareDataset(data)
	images, labels = ShuffleData(images, labels)
	encodedLabels := ToOneHot(labels, 10)

	// Split the dataset into training and testing sets
	trainSize := int(0.8 * float64(len(images))) 

	trainImages := images[:trainSize]
	trainLabels := encodedLabels[:trainSize]
	testImages := images[trainSize:]
	testLabels := encodedLabels[trainSize:]

	// Create batches for training
	// imagesBatch, labelsBatches := CreateBatches(trainImages, trainLabels, 128)

	// _testImages, _testLabels := CreateBatches(testImages,testLabels, 128)

	// Initialize neural network
	randomizer := rand.New(rand.NewSource(time.Now().UnixNano()))
	nn := NewNN(0.01, 50, Sigmoid, CrossEntropy_T) 
	nn.AddLayer(784, 128, randomizer, Sigmoid)
	nn.AddLayer(128, 10, randomizer, Softmax)

	nn.Train(trainImages, trainLabels)

	testAccuracy := nn.CalculateAccuracy(testImages, testLabels)
	fmt.Printf("Test Accuracy: %f\n", testAccuracy)

	nn.SaveModel("model.json")
}