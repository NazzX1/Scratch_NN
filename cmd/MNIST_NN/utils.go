package main



func GetMaxIndex(arr []float64) int {
    maxIndex := 0
    for i, value := range arr {
        if value > arr[maxIndex] {
            maxIndex = i
        }
    }
    return maxIndex
}