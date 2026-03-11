package main

import (
	"encoding/json"
	"fmt"
	"net/http"
)

func handleSolve(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"success":false,"error":"method not allowed, use POST"}`, http.StatusMethodNotAllowed)
		return
	}

	var input ProblemInput
	if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ProblemOutput{
			Success: false,
			Error:   fmt.Sprintf("invalid JSON: %v", err),
		})
		return
	}

	result := solveProblem(input)

	w.Header().Set("Content-Type", "application/json")
	if !result.Success {
		w.WriteHeader(http.StatusUnprocessableEntity)
	}
	json.NewEncoder(w).Encode(result)
}

func handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"status":"ok","service":"go-fem"}`))
}

func handleInfo(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"service": "go-fem",
		"version": "0.2.0",
		"elements": []string{
			"tet4", "hexa8", "tet10", "brick20",
			"truss3d", "corot_truss",
			"elastic_beam3d",
			"quad4",
			"shell_mitc4", "dkt3",
			"zerolength",
		},
		"materials": []string{"isotropic_linear", "orthotropic_linear"},
		"solvers":   []string{"cholesky", "lu"},
		"endpoints": map[string]string{
			"POST /solve": "Submit a FEM problem (JSON) and get displacement results",
			"GET /health": "Health check",
			"GET /":       "This info page",
		},
	})
}
