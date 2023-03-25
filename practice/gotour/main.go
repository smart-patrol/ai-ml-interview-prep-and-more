package main

import (
	"fmt"
	"strings"
)

func WordCount(s string) map[string]int {
	words := strings.Fields(s)
	m := make(map[string]int)
	for _, word := range words {
		m[word] += 1
	}
	return m
}

func main() {
	fmt.Println(WordCount("I am learning go."))
	// This has 1 word.
	fmt.Println(WordCount("The quick brown fox jumped over the lazy dog."))
	// This has 2 words.
	fmt.Println(WordCount("A man a plan a canal panama"))
}

// import (
// 	"io"
// 	"log"
// 	"net/http"
// )

// // https://www.wolfe.id.au/2020/03/10/starting-a-go-project/
// func main() {
// 	// Hello world, the web server

// 	helloHandler := func(w http.ResponseWriter, req *http.Request) {
// 		io.WriteString(w, "Hello, world!\n")
// 	}

// 	http.HandleFunc("/hello", helloHandler)
// 	log.Println("Listing for requests at http://localhost:8000/hello")
// 	log.Fatal(http.ListenAndServe(":8000", nil))
// }
