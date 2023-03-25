package main

import (
	"fmt"
	"math/rand"
	"time"
)

func SelectionSort(arr []int){
	var n = len(arr)
	for i:=1; i<n; i++{
		min_idx := i
		for j:=i+1; j<n; j++{
			if arr[j] <arr[min_idx]:
			min_idx =j
		}
	// swap the found minimum
	arr[i], arr[min_idx] = arr[min_idx], arr[i]
	}
}