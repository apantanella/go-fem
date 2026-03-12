// go-fem HTTP API server.
//
// Exposes a JSON API that accepts a FEM problem definition
// and returns displacement results.
//
// Usage:
//
//	go run ./cmd/femgo                  # listen on :8080
//	go run ./cmd/femgo -addr :9090      # custom port
//	curl -X POST http://localhost:8080/solve -d @problem.json
package main

import (
	"flag"
	"log"
	"net/http"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	flag.Parse()

	mux := http.NewServeMux()
	mux.HandleFunc("/", handleInfo)
	mux.HandleFunc("/health", handleHealth)
	mux.HandleFunc("/solve", handleSolve)
	mux.HandleFunc("/elements", handleElements)

	log.Printf("go-fem API server listening on %s", *addr)
	log.Printf("  POST /solve    – solve a FEM problem")
	log.Printf("  GET  /elements – list elements by dimension")
	log.Printf("  GET  /health   – health check")
	log.Printf("  GET  /         – API info")
	log.Fatal(http.ListenAndServe(*addr, mux))
}
