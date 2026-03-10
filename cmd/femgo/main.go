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
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"time"

	"go-fem/analysis"
	"go-fem/domain"
	"go-fem/element"
	"go-fem/material"
	"go-fem/solver"
)

// ---------------------------------------------------------------------------
// JSON input types
// ---------------------------------------------------------------------------

type ProblemInput struct {
	Materials          []MaterialInput `json:"materials"`
	Nodes              [][3]float64    `json:"nodes"`
	Elements           []ElementInput  `json:"elements"`
	BoundaryConditions []BCInput       `json:"boundary_conditions"`
	Loads              []LoadInput     `json:"loads"`
	Solver             string          `json:"solver,omitempty"` // "cholesky" (default) | "lu"
}

type MaterialInput struct {
	ID   string  `json:"id"`
	Type string  `json:"type"` // "isotropic_linear"
	E    float64 `json:"E"`
	Nu   float64 `json:"nu"`
}

type ElementInput struct {
	Type     string `json:"type"` // "tet4" | "hexa8"
	Material string `json:"material"`
	Nodes    []int  `json:"nodes"`
}

type BCInput struct {
	Node int   `json:"node"`
	DOFs []int `json:"dofs"` // e.g. [0,1,2] for all, [2] for Z only
}

type LoadInput struct {
	Node  int     `json:"node"`
	DOF   int     `json:"dof"` // 0=X, 1=Y, 2=Z
	Value float64 `json:"value"`
}

// ---------------------------------------------------------------------------
// JSON output types
// ---------------------------------------------------------------------------

type ProblemOutput struct {
	Success       bool                 `json:"success"`
	Error         string               `json:"error,omitempty"`
	Info          *InfoOutput          `json:"info,omitempty"`
	Displacements []DisplacementOutput `json:"displacements,omitempty"`
	Summary       *SummaryOutput       `json:"summary,omitempty"`
	ElapsedMs     float64              `json:"elapsed_ms,omitempty"`
}

type InfoOutput struct {
	NumNodes    int    `json:"num_nodes"`
	NumElements int    `json:"num_elements"`
	NumDOFs     int    `json:"num_dofs"`
	Solver      string `json:"solver"`
}

type DisplacementOutput struct {
	Node int     `json:"node"`
	Ux   float64 `json:"ux"`
	Uy   float64 `json:"uy"`
	Uz   float64 `json:"uz"`
}

type SummaryOutput struct {
	MaxAbsDisplacement MaxDispOutput `json:"max_abs_displacement"`
}

type MaxDispOutput struct {
	Node      int     `json:"node"`
	Component string  `json:"component"` // "ux", "uy", or "uz"
	Value     float64 `json:"value"`
}

// ---------------------------------------------------------------------------
// Solver logic
// ---------------------------------------------------------------------------

func solveProblem(input ProblemInput) ProblemOutput {
	t0 := time.Now()

	// --- Build materials map ---
	mats := make(map[string]material.Material3D, len(input.Materials))
	for _, mi := range input.Materials {
		switch mi.Type {
		case "isotropic_linear":
			mats[mi.ID] = material.NewIsotropicLinear(mi.E, mi.Nu)
		default:
			return errorResponse("unknown material type: %s", mi.Type)
		}
	}

	// --- Build domain ---
	dom := domain.NewDomain()

	for _, n := range input.Nodes {
		dom.AddNode(n[0], n[1], n[2])
	}

	for eid, ei := range input.Elements {
		mat3d, ok := mats[ei.Material]
		if !ok {
			return errorResponse("element %d references unknown material: %s", eid, ei.Material)
		}

		switch ei.Type {
		case "tet4":
			if len(ei.Nodes) != 4 {
				return errorResponse("element %d (tet4) requires 4 nodes, got %d", eid, len(ei.Nodes))
			}
			var nodes [4]int
			var coords [4][3]float64
			for i, nid := range ei.Nodes {
				if nid < 0 || nid >= len(dom.Nodes) {
					return errorResponse("element %d: node %d out of range", eid, nid)
				}
				nodes[i] = nid
				coords[i] = dom.Nodes[nid].Coord
			}
			dom.AddElement(element.NewTet4(eid, nodes, coords, mat3d))

		case "hexa8":
			if len(ei.Nodes) != 8 {
				return errorResponse("element %d (hexa8) requires 8 nodes, got %d", eid, len(ei.Nodes))
			}
			var nodes [8]int
			var coords [8][3]float64
			for i, nid := range ei.Nodes {
				if nid < 0 || nid >= len(dom.Nodes) {
					return errorResponse("element %d: node %d out of range", eid, nid)
				}
				nodes[i] = nid
				coords[i] = dom.Nodes[nid].Coord
			}
			dom.AddElement(element.NewHexa8(eid, nodes, coords, mat3d))

		default:
			return errorResponse("unknown element type: %s", ei.Type)
		}
	}

	// --- Boundary conditions ---
	for _, bc := range input.BoundaryConditions {
		if bc.Node < 0 || bc.Node >= len(dom.Nodes) {
			return errorResponse("BC: node %d out of range", bc.Node)
		}
		for _, dof := range bc.DOFs {
			if dof < 0 || dof > 2 {
				return errorResponse("BC: invalid dof %d (must be 0, 1, or 2)", dof)
			}
			dom.FixDOF(bc.Node, dof)
		}
	}

	// --- Loads ---
	for _, ld := range input.Loads {
		if ld.Node < 0 || ld.Node >= len(dom.Nodes) {
			return errorResponse("load: node %d out of range", ld.Node)
		}
		if ld.DOF < 0 || ld.DOF > 2 {
			return errorResponse("load: invalid dof %d", ld.DOF)
		}
		dom.ApplyLoad(ld.Node, ld.DOF, ld.Value)
	}

	// --- Pick solver ---
	solverName := "cholesky"
	var slv solver.LinearSolver
	switch input.Solver {
	case "", "cholesky":
		slv = solver.Cholesky{}
	case "lu":
		slv = solver.LU{}
		solverName = "lu"
	default:
		return errorResponse("unknown solver: %s", input.Solver)
	}

	// --- Solve ---
	ana := analysis.StaticLinearAnalysis{Dom: dom, Solver: slv}
	U, err := ana.Run()
	if err != nil {
		return errorResponse("analysis failed: %v", err)
	}

	disp := dom.SetDisplacements(U)

	// --- Build displacement output ---
	disps := make([]DisplacementOutput, len(disp))
	var maxVal float64
	var maxNode int
	var maxComp string
	for i, d := range disp {
		disps[i] = DisplacementOutput{Node: i, Ux: d[0], Uy: d[1], Uz: d[2]}
		for c, v := range d {
			if math.Abs(v) > math.Abs(maxVal) {
				maxVal = v
				maxNode = i
				maxComp = [3]string{"ux", "uy", "uz"}[c]
			}
		}
	}

	elapsed := time.Since(t0).Seconds() * 1000

	return ProblemOutput{
		Success: true,
		Info: &InfoOutput{
			NumNodes:    len(dom.Nodes),
			NumElements: len(dom.Elements),
			NumDOFs:     dom.NumDOF(),
			Solver:      solverName,
		},
		Displacements: disps,
		Summary: &SummaryOutput{
			MaxAbsDisplacement: MaxDispOutput{
				Node:      maxNode,
				Component: maxComp,
				Value:     maxVal,
			},
		},
		ElapsedMs: elapsed,
	}
}

func errorResponse(format string, args ...any) ProblemOutput {
	return ProblemOutput{
		Success: false,
		Error:   fmt.Sprintf(format, args...),
	}
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

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
		"service":  "go-fem",
		"version":  "0.1.0",
		"elements": []string{"tet4", "hexa8"},
		"materials": []string{"isotropic_linear"},
		"solvers":  []string{"cholesky", "lu"},
		"endpoints": map[string]string{
			"POST /solve": "Submit a FEM problem (JSON) and get displacement results",
			"GET /health":  "Health check",
			"GET /":        "This info page",
		},
	})
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	flag.Parse()

	mux := http.NewServeMux()
	mux.HandleFunc("/", handleInfo)
	mux.HandleFunc("/health", handleHealth)
	mux.HandleFunc("/solve", handleSolve)

	log.Printf("go-fem API server listening on %s", *addr)
	log.Printf("  POST /solve   – solve a FEM problem")
	log.Printf("  GET  /health  – health check")
	log.Printf("  GET  /        – API info")
	log.Fatal(http.ListenAndServe(*addr, mux))
}
