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
			// 3D
			"tet4_3d", "hexa8_3d", "tet10_3d", "brick20_3d",
			"truss_3d", "corot_truss_3d",
			"elastic_beam_3d", "timoshenko_beam_3d",
			"shell_mitc4_3d", "dkt3_3d",
			"zerolength_3d", "zerolength_trans_3d",
			// 2D
			"truss_2d",
			"elastic_beam_2d", "timoshenko_beam_2d",
			"quad4_2d", "quad8_2d", "tri3_2d", "tri6_2d",
			"zerolength_2d", "zerolength_frame_2d",
		},
		"materials": []string{"isotropic_linear", "orthotropic_linear"},
		"solvers":   []string{"cholesky", "lu"},
		"endpoints": map[string]string{
			"POST /solve":    "Submit a FEM problem (JSON) and get displacement results",
			"GET  /elements": "List available elements grouped by dimension",
			"GET  /health":   "Health check",
			"GET  /":         "This info page",
		},
	})
}

func handleElements(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"2D": []string{
			"truss_2d",
			"elastic_beam_2d",
			"timoshenko_beam_2d",
			"quad4_2d",
			"quad8_2d",
			"tri3_2d",
			"tri6_2d",
			"zerolength_2d",
			"zerolength_frame_2d",
		},
		"3D": []string{
			"tet4_3d",
			"hexa8_3d",
			"tet10_3d",
			"brick20_3d",
			"truss_3d",
			"corot_truss_3d",
			"elastic_beam_3d",
			"timoshenko_beam_3d",
			"shell_mitc4_3d",
			"dkt3_3d",
			"zerolength_3d",
			"zerolength_trans_3d",
		},
	})
}
