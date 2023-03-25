package main

import "fmt"

func main() {
	unsorted := []int{10, 6, 2, 1, 5, 8, 3, 4, 7, 9}
	sorted := bubbleSort(unsorted)
	fmt.Println(sorted)
	// sorted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

func bubbleSort(arr []int) []int {
	swapped := true
	for swapped {
		swapped = false
		for i := 1; i < len(arr); i++ {
			if arr[i-1] > arr[i] {
				arr[i], arr[i-1] = arr[i-1], arr[i]
				swapped = true
			}
		}
	}
	return arr
}
