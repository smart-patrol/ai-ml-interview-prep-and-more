package main

import (
	"fmt"
	"math/rand"
	"time"
)

func main() {

	slice := generateSlice(20)
	fmt.Println("\n--- Unsorted --- \n\n", slice)
	quicksort(slice)
	fmt.Println("\n--- Sorted ---\n\n", slice, "\n")
}

// Generates a slice of size, size filled with random numbers
func generateSlice(size int) []int {

	slice := make([]int, size, size)
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < size; i++ {
		slice[i] = rand.Intn(999) - rand.Intn(999)
	}
	return slice
}

func quicksort(arr []int) []int {
	if len(arr) < 2 {
		return arr
	}
	left, right := 0, len(arr)-1
	pivot := rand.Int() % len(arr)
	arr[pivot], arr[right] = arr[right], arr[pivot]

	for i, _ := range arr {
		if arr[i] < arr[right] {
			arr[left], arr[i] = arr[i], arr[left]
			left++
		}
	}
	arr[left], arr[right] = arr[right], arr[left]

	quicksort(arr[:left])
	quicksort(arr[left+1:])
	return arr
}
