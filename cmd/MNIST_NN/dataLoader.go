package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"golang.org/x/tools/go/analysis/passes/nilness"
)

type DataPoint struct {
	inputs []float64
	labels int
}

func LoadDataset(path string) [][]string{
	file, err := os.Open(path)

	if err != nil {
		log.Fatal(err)
	}

	defer file.Close()
	reader := csv.NewReader(file)

	data, err := reader.ReadAll()

	if err != nil {
		log.Fatal(err)
	}

	defer fmt.Println("Your data has been loaded successfully !!!")
	return data
}

func PrepareDataset(data [][]string) ([][]float64, []int){
	var images [][]float64
	var labels []int

	for i, j:= range data{

		if i == 0{ //skip the header
			continue
		}
		label, err := strconv.Atoi(j[0])
		if err != nil{
			log.Fatal(err)
		}
		labels = append(labels, label)
	

		var image []float64
		for _, pixel := range j[1:]{
			pixelValue, err := strconv.Atoi(pixel)
			if err != nil {
				log.Fatal(err)
			}
			image = append(image, float64(pixelValue)/250) // Normalize it to [0, 1]
		}
		images = append(images, image)
	}
	return images, labels
}

func ShuffleData(images [][]float64, labels []int) ([][]float64, []int){
	
	for i := range images{
		j := rand.Intn(len(images))
		images[i], images[j] = images[j], images[i]
		labels[i], labels[j] = labels[j], labels[i] 
	}

	return images, labels
	
}

// TODO: create batches and then go back to fix the bias gradient

func CreateBatches(images [][]float64, labels []int, batchSize int){

}
