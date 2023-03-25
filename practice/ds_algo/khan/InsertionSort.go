package main

import "fmt"

// insertion sort implementation in Go
func InsertionSort(arr []int) int {

	var n = len(arr)
	for i := 1; i < n; i++ {
		j := i
		for j > 0 {
			if arr[j-1] > arr[j] {
				arr[j-1], arr[j] = arr[j], arr[j-1]
			}
			j = j - 1
		}
	}
}

func main() {
	x := []int{10, 20, 6, 2, 2}
	InsertionSort(x)
	fmt.Println(x)
}
