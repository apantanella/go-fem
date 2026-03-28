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
		"endpoints": []map[string]string{
			{"method": "POST", "path": "/solve", "description": "Submit a FEM problem (JSON) and get displacement results"},
			{"method": "GET", "path": "/solve-example", "description": "Example problem with complete structure (2D cantilever beam)"},
			{"method": "GET", "path": "/elements", "description": "List available elements grouped by dimension"},
			{"method": "GET", "path": "/nodes", "description": "Node structure and examples"},
			{"method": "GET", "path": "/materials", "description": "Material types and examples"},
			{"method": "GET", "path": "/loads", "description": "Load types and examples"},
			{"method": "GET", "path": "/boundary-conditions", "description": "Constraint types and examples"},
			{"method": "GET", "path": "/sections", "description": "Section properties and examples"},
			{"method": "GET", "path": "/health", "description": "Health check"},
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

func handleNodes(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"description": "Nodes define the coordinates of mesh points in the FEM model",
		"format": "array of [x, y, z] coordinates in global reference frame",
		"units": "use consistent units (e.g. mm, m)",
		"indexing": "0-based: first node is index 0, used in element and BC definitions",
		"example": []any{
			map[string]any{
				"description": "4 nodes forming a square in 2D (z=0)",
				"nodes": [][3]float64{
					{0, 0, 0},
					{100, 0, 0},
					{100, 100, 0},
					{0, 100, 0},
				},
			},
			map[string]any{
				"description": "3D tetrahedral mesh with 4 vertices",
				"nodes": [][3]float64{
					{0, 0, 0},
					{100, 0, 0},
					{0, 100, 0},
					{0, 0, 100},
				},
			},
		},
	})
}

func handleMaterials(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"description": "Materials define the constitutive properties of elements",
		"types": []map[string]any{
			{
				"type": "isotropic_linear",
				"description": "Linear isotropic elastic material (most common)",
				"required_fields": []string{"id", "type", "E", "nu"},
				"example": map[string]any{
					"id":   "steel",
					"type": "isotropic_linear",
					"E":    210000,  // MPa (Young's modulus)
					"nu":   0.3,     // Poisson's ratio
				},
			},
			{
				"type": "orthotropic_linear",
				"description": "Linear orthotropic elastic material (composites, wood)",
				"required_fields": []string{"id", "type", "Ex", "Ey", "Ez", "Gxy", "Gyz", "Gxz", "nxy", "nyz", "nxz"},
				"example": map[string]any{
					"id":   "composite",
					"type": "orthotropic_linear",
					"Ex":   150000,
					"Ey":   10000,
					"Ez":   10000,
					"Gxy":  5000,
					"Gyz":  5000,
					"Gxz":  5000,
					"nxy":  0.25,
					"nyz":  0.3,
					"nxz":  0.3,
				},
			},
			{
				"type": "steel_bilinear",
				"description": "Bilinear elasto-plastic steel material",
				"required_fields": []string{"id", "type", "E", "nu", "Fy", "Esh"},
				"example": map[string]any{
					"id":   "steel_plastic",
					"type": "steel_bilinear",
					"E":    210000,
					"nu":   0.3,
					"Fy":   355,      // yield stress (MPa)
					"Esh":  2100,     // hardening modulus (MPa)
				},
			},
			{
				"type": "concrete_pararect",
				"description": "Parabola-rectangle concrete (EN 1992-1-1)",
				"required_fields": []string{"id", "type", "Fc"},
				"optional_fields": []string{"EpsC1", "EpsCU"},
				"example": map[string]any{
					"id":       "concrete_c30",
					"type":     "concrete_pararect",
					"Fc":       30,       // compressive strength (MPa)
					"eps_c1":   0.002,    // strain at peak
					"eps_cu":   0.0035,   // ultimate strain
				},
			},
		},
	})
}

func handleLoads(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"description": "Loads apply forces, moments, pressures, and body forces to the model",
		"types": []map[string]any{
			{
				"type": "nodal",
				"description": "Concentrated force or moment at a node (default if type omitted)",
				"fields": map[string]string{
					"node":  "node index (0-based)",
					"dof":   "degree of freedom: 0=UX, 1=UY, 2=UZ, 3=RX, 4=RY, 5=RZ",
					"value": "force (N) or moment (N⋅mm)",
				},
				"example": map[string]any{
					"node":  0,
					"dof":   2,      // vertical load in Z direction
					"value": 1000,   // 1000 N downward
				},
			},
			{
				"type": "surface_pressure",
				"description": "Uniform pressure on a 4-node face",
				"fields": map[string]string{
					"type":       "surface_pressure",
					"face_nodes": "4 global node indices (CCW from outside)",
					"pressure":   "pressure value (Pa or MPa)",
				},
				"example": map[string]any{
					"type":       "surface_pressure",
					"face_nodes": [4]int{0, 1, 2, 3},
					"pressure":   0.01,  // 0.01 MPa
				},
			},
			{
				"type": "beam_dist",
				"description": "Uniform distributed load on a beam element",
				"fields": map[string]string{
					"type":      "beam_dist",
					"element":   "element index",
					"dir":       "global direction unit vector [x, y, z]",
					"intensity": "load per unit length (N/mm or N/m)",
				},
				"example": map[string]any{
					"type":      "beam_dist",
					"element":   0,
					"dir":       [3]float64{0, -1, 0},
					"intensity": 10,  // 10 N/mm
				},
			},
			{
				"type": "body_force",
				"description": "Body force (gravity) on a solid element",
				"fields": map[string]string{
					"type": "body_force",
					"element": "element index",
					"rho":  "mass density (kg/mm³)",
					"g":    "gravitational acceleration [x, y, z] (mm/s²)",
				},
				"example": map[string]any{
					"type":    "body_force",
					"element": 0,
					"rho":     0.0000078,  // steel: 7800 kg/m³ = 7.8e-6 kg/mm³
					"g":       [3]float64{0, -9.81e6, 0},  // gravity in mm/s²
				},
			},
		},
	})
}

func handleBoundaryConditions(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"description": "Boundary conditions (BCs) constrain degrees of freedom at nodes",
		"dof_mapping": map[string]int{
			"0": 0,  // UX – translation X
			"1": 1,  // UY – translation Y
			"2": 2,  // UZ – translation Z
			"3": 3,  // RX – rotation X
			"4": 4,  // RY – rotation Y
			"5": 5,  // RZ – rotation Z
		},
		"constraint_types": []map[string]any{
			{
				"name": "Fixed support",
				"description": "Fully fixed: all translational and rotational DOFs restrained",
				"example": map[string]any{
					"node": 0,
					"dofs": []int{0, 1, 2, 3, 4, 5},
				},
			},
			{
				"name": "Pinned support",
				"description": "Translations fixed, rotations free",
				"example": map[string]any{
					"node": 0,
					"dofs": []int{0, 1, 2},
				},
			},
			{
				"name": "Roller support (horizontal)",
				"description": "Vertical translation fixed, horizontal free",
				"example": map[string]any{
					"node": 5,
					"dofs": []int{1},  // constrain Y only
				},
			},
			{
				"name": "Prescribed displacement",
				"description": "Apply non-zero displacement or rotation at a DOF",
				"example": map[string]any{
					"node":   10,
					"dofs":   []int{2},
					"values": []float64{5.0},  // 5 mm downward settlement
				},
			},
			{
				"name": "Symmetry constraint (vertical axis)",
				"description": "Constrain horizontal motion perpendicular to symmetry plane",
				"example": map[string]any{
					"node": 15,
					"dofs": []int{0},  // constrain UX on vertical symmetry plane
				},
			},
		},
	})
}

func handleSections(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"description": "Sections define geometric and cross-sectional properties",
		"usage": "Reference sections by ID in elements; useful for beams, shells, and frame members",
		"properties": []map[string]any{
			{
				"name": "Beam section",
				"fields": map[string]string{
					"A":         "cross-sectional area (mm²)",
					"Iy":        "moment of inertia about Y (mm⁴)",
					"Iz":        "moment of inertia about Z (mm⁴)",
					"J":         "torsional constant (mm⁴)",
					"Asy":       "effective shear area Y (mm²)",
					"Asz":       "effective shear area Z (mm²)",
				},
				"example": map[string]any{
					"id": "IPE240",
					"A":  3910,
					"Iy": 3817000,
					"Iz": 284000,
					"J":  35200,
				},
			},
			{
				"name": "Shell/Plate section",
				"fields": map[string]string{
					"thickness": "uniform thickness (mm)",
				},
				"example": map[string]any{
					"id":        "slab_200",
					"thickness": 200,
				},
			},
		},
	})
}

func handleSolveExample(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	// Return only the problem data, ready to POST directly to /solve
	json.NewEncoder(w).Encode(map[string]any{
		"dimensions": "2D",
		"analysis_type": "static",
		"materials": []map[string]any{
			{
				"id":   "steel",
				"type": "isotropic_linear",
				"E":    210000,  // Young's modulus (MPa)
				"nu":   0.3,     // Poisson's ratio
			},
		},
		"sections": []map[string]any{
			{
				"id": "beam_section",
				"A":  5000,       // 50mm × 100mm
				"Iy": 41666667,   // bh³/12 = 100×50³/12
			},
		},
		"nodes": [][3]float64{
			{0, 0, 0},       // Node 0: fixed support (left)
			{500, 0, 0},     // Node 1
			{1000, 0, 0},    // Node 2: free end (right)
		},
		"elements": []map[string]any{
			{
				"type":    "elastic_beam_2d",
				"material": "steel",
				"section":  "beam_section",
				"nodes":    []int{0, 1},
			},
			{
				"type":    "elastic_beam_2d",
				"material": "steel",
				"section":  "beam_section",
				"nodes":    []int{1, 2},
			},
		},
		"boundary_conditions": []map[string]any{
			{
				"node": 0,
				"dofs": []int{0, 1, 2},  // Fixed: UX, UY, RZ
			},
		},
		"loads": []map[string]any{
			{
				"type":      "beam_dist",
				"element":   0,
				"dir":       [3]float64{0, -1, 0},  // Downward
				"intensity": 10,                     // 10 N/mm
			},
			{
				"type":      "beam_dist",
				"element":   1,
				"dir":       [3]float64{0, -1, 0},  // Downward
				"intensity": 10,                     // 10 N/mm
			},
		},
		"solver": "lu",
	})
}
