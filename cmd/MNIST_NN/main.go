package main

import (
	"fmt"
	"math/rand"
	"time"
)

// Assuming LoadDataset, PrepareDataset, ShuffleData, ToOneHot, CreateBatches, and GetMaxIndex are already defined

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
	nn := NewNN(0.01, 20, Sigmoid, CrossEntropy_T) 
	nn.AddLayer(784, 128, randomizer)
	nn.AddLayer(128, 64, randomizer)
	nn.AddLayer(64, 10, randomizer)

	// Train the neural network
	nn.Train(trainImages, trainLabels)

	// Evaluate the neural network on the test set
	testAccuracy := nn.CalculateAccuracy(testImages, testLabels)
	fmt.Printf("Test Accuracy: %f\n", testAccuracy)
}