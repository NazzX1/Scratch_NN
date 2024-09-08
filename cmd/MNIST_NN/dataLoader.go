package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
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

	defer fmt.Println("your data has been loaded successfully !!!")
	return data
}


func main()  {
	
	LoadDataset("../../data/mnist_train.csv")
}